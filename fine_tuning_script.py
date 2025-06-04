import os
import json
import argparse
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from evaluate import load # Importa a biblioteca evaluate para metricas
import jsonlines # Adicionado para leitura robusta de JSONL

# --- 1. Definicao do Dataset Personalizado ---
class MedicalCertificateDataset(Dataset):
    """
    Dataset personalizado para carregar imagens de atestados medicos e seus textos alvo.
    """
    def __init__(self, data_path, processor, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        """
        Carrega os dados de um arquivo JSONL (JSON Lines) usando a biblioteca jsonlines.
        Cada linha do arquivo deve ser um objeto JSON.
        """
        data = []
        if not os.path.exists(data_path):
            print(f"Erro: O arquivo de dados nao foi encontrado em {data_path}. Verifique o caminho.")
            return []
        try:
            # Usa jsonlines.open para ler o arquivo JSONL de forma mais robusta
            with jsonlines.open(data_path, 'r') as reader:
                for obj in reader:
                    data.append(obj)
            print(f"Dados carregados com sucesso de {data_path}. Total de {len(data)} itens.")
        except jsonlines.InvalidLineError as e:
            print(f"Aviso: Linha invalida no arquivo {data_path}: {e}. Pulando linha.")
        except Exception as e:
            print(f"Erro inesperado ao ler o arquivo {data_path}: {e}")
            return []
        return data

    def __len__(self):
        """Retorna o numero total de itens no dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retorna um item do dataset (imagem processada e labels de texto).
        """
        item = self.data[idx]
        image_path = item["image"]
        target_text = item["target_text"]

        # Verifica se o caminho da imagem e absoluto ou relativo
        # Para o Docker, certifique-se de que o diretorio de imagens seja montado corretamente.
        # Ex: se o caminho no JSON for 'images/atestado1.jpg' e 'images' esta na raiz do projeto
        # ele sera encontrado.
        # Nao e necessario modificar o image_path aqui se os volumes Docker estiverem configurados corretamente.

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Aviso: Imagem nao encontrada em {image_path}. Pulando este item.")
            return None # Retorna None para que o DataLoader possa lidar com isso
        except Exception as e:
            print(f"Erro ao abrir a imagem {image_path}: {e}. Pulando este item.")
            return None

        # Processar a imagem usando o feature extractor do TrOCR
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Processar o texto target usando o tokenizer do TrOCR
        labels = self.processor.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Decodificador TrOCR espera que os labels sejam -100 para tokens de padding
        # Isso garante que a perda nao seja calculada sobre os tokens de padding.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values.squeeze(), "labels": labels.squeeze()}

# Funcao para filtrar itens None do DataLoader
def collate_fn(batch):
    """
    Funcao de colagem para o DataLoader, que filtra itens None (imagens nao encontradas ou com erro).
    Retorna um dicionario vazio se o batch estiver vazio apos a filtragem,
    para evitar TypeError no Trainer.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Se o batch estiver vazio, retorna um dicionario vazio.
        # O Trainer pode lidar com isso, pulando este passo de treinamento.
        return {} 

    # Empilha os tensores de pixel_values e labels
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels}

# Inicializa a metrica CER (Character Error Rate)
# Esta metrica sera usada na funcao compute_metrics
cer_metric = load("cer")

def compute_metrics(pred):
    """
    Funcao para calcular metricas de avaliacao durante o treinamento.
    Calcula o Character Error Rate (CER).
    """
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decodifica as previsoes do modelo para texto
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Substitui -100 (tokens de padding ignorados na perda) pelo token de padding real
    # para que o tokenizer possa decodifica-los corretamente.
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Calcula o CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

# --- 2. Funcao Principal de Fine-tuning ---
def run_fine_tuning(
    data_json_path,
    model_output_dir="fine_tuned_trocr_model",
    epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    save_total_limit=3, # Limita o numero total de checkpoints salvos
    gradient_accumulation_steps=1, # NOVO: Adicionado para controlar a acumulacao de gradientes
):
    """
    Executa o processo de fine-tuning do modelo TrOCR.

    Args:
        data_json_path (str): Caminho para o arquivo JSONL contendo os dados de treinamento.
        model_output_dir (str): Diretorio para salvar o modelo e o processador treinados.
        epochs (int): Numero de epocas de treinamento.
        batch_size (int): Tamanho do batch para treinamento e avaliacao.
        learning_rate (float): Taxa de aprendizado para o otimizador.
        eval_steps (int): Numero de passos entre cada avaliacao.
        save_steps (int): Numero de passos entre cada salvamento do modelo.
        logging_steps (int): Numero de passos entre cada log de treinamento.
        save_total_limit (int): Limite o numero total de checkpoints a serem salvos.
        gradient_accumulation_steps (int): Numero de passos para acumular gradientes antes de atualizar os pesos.
    """
    global processor # Declara processor como global para ser acessivel em compute_metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Carregar processador e modelo pre-treinado da Microsoft
    print("Carregando processador e modelo pre-treinado 'microsoft/trocr-base-handwritten'...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Configurar o decodificador para geracao de texto
    # Estes sao parametros importantes para a geracao de texto do modelo.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Criar dataset personalizado
    print(f"Carregando dados do dataset de: {data_json_path}")
    dataset = MedicalCertificateDataset(data_json_path, processor)

    if not dataset or len(dataset) == 0: # Adicionada verificacao de dataset vazio
        print("Nenhum dado valido encontrado no dataset. Abortando treinamento.")
        return

    # Dividir o dataset em treino e validacao (90% treino, 10% validacao)
    # Garante que haja dados suficientes para ambos os datasets
    if len(dataset) < 2: # Minimo de 2 para divisao (1 para treino, 1 para validacao)
        print("Dataset muito pequeno para divisao em treino/validacao. Abortando treinamento.")
        return

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Garante que ambos os tamanhos sejam pelo menos 1
    # Ajusta o train_size se o dataset for muito pequeno para garantir val_size > 0
    if train_size == 0: train_size = 1
    if val_size == 0: val_size = 1
    if train_size + val_size > len(dataset): # Se a soma exceder o total (devido a arredondamento ou dataset pequeno)
        train_size = len(dataset) - val_size # Ajusta o treino para o restante

    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Tamanho do dataset de treino: {len(train_dataset)} amostras.")
    print(f"Tamanho do dataset de avaliacao: {len(eval_dataset)} amostras.")

    # Configurar argumentos de treinamento
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="steps", # Avalia a cada 'eval_steps' passos
        eval_steps=eval_steps,
        save_strategy="steps", # Salva o checkpoint a cada 'save_steps' passos
        save_steps=save_steps,
        save_total_limit=save_total_limit, # Limita o numero de checkpoints salvos
        logging_steps=logging_steps, # Loga metricas a cada 'logging_steps' passos
        logging_dir="./logs", # Diretorio para logs (ex: para TensorBoard)
        report_to="none", # Desabilita integracao com Weights & Biases, etc.
        load_best_model_at_end=True, # Carrega o melhor modelo (baseado em eval_loss) no final do treinamento
        metric_for_best_model="eval_loss", # A metrica principal para determinar o "melhor" modelo
        greater_is_better=False, # Para loss, menor e melhor
        fp16=True if device == "cuda" else False, # Habilita Mixed Precision Training para GPU (mais rapido, menos memoria)
        dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 0, # Otimiza o carregamento de dados
        gradient_accumulation_steps=gradient_accumulation_steps, # NOVO: Adicionado aqui
    )

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn, # Usa a funcao de colagem personalizada
        compute_metrics=compute_metrics, # Adiciona a funcao para calcular metricas
    )

    # Treinar o modelo
    print("Iniciando treinamento do modelo TrOCR...")
    trainer.train()
    print("Treinamento concluido!")

    # Salvar o modelo e o processador treinados
    print(f"Salvando modelo e processador fine-tuned em {model_output_dir}...")
    os.makedirs(model_output_dir, exist_ok=True) # Garante que o diretorio exista
    processor.save_pretrained(model_output_dir)
    model.save_pretrained(model_output_dir)
    print("Modelo e processador salvos com sucesso.")

    # --- Exemplo de Inferência (Apos o treinamento) ---
    print("\n--- Exemplo de Inferencia apos o treinamento ---")
    if eval_dataset and len(eval_dataset) > 0:
        try:
            # Pega um item do dataset de validacao para teste
            sample_item = eval_dataset[0]
            # Adiciona dimensao de batch e move para o dispositivo correto
            pixel_values_sample = sample_item["pixel_values"].unsqueeze(0).to(device)

            # Gera o texto a partir da imagem
            generated_ids = model.generate(pixel_values_sample)
            generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Decodifica o texto original (label) para comparacao
            original_text = processor.tokenizer.decode(
                sample_item['labels'][sample_item['labels'] != -100],
                skip_special_tokens=True
            )
            print(f"Texto Original (Label): {original_text}")
            print(f"Texto Gerado (Previsao): {generated_text}")
        except IndexError:
            print("Nenhum dado de avaliacao disponivel para exemplo de inferencia.")
        except Exception as e:
            print(f"Erro durante o exemplo de inferencia: {e}")
    else:
        print("Nenhum dataset de avaliacao disponivel para exemplo de inferencia.")


if __name__ == "__main__":
    # Configuracao do parser de argumentos para permitir que o Streamlit passe parametros
    parser = argparse.ArgumentParser(description="Script de Fine-tuning do TrOCR para Atestados Medicos.")
    parser.add_argument("--data_json_path", type=str, required=True,
                        help="Caminho para o arquivo JSONL contendo os dados de treinamento.")
    parser.add_argument("--model_output_dir", type=str, default="fine_tuned_trocr_model",
                        help="Diretorio para salvar o modelo e o processador treinados.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Numero de epocas de treinamento.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Tamanho do batch para treinamento e avaliacao.")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Taxa de aprendizado para o otimizador.")
    parser.add_argument("--save_total_limit", type=int, default=3, # Adicionado este argumento
                        help="Limite o numero total de checkpoints a serem salvos.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, # NOVO: Adicionado este argumento
                        help="Numero de passos para acumular gradientes antes de atualizar os pesos.")


    args = parser.parse_args()

    # Rodar o fine-tuning com os argumentos fornecidos
    run_fine_tuning(
        data_json_path=args.data_json_path,
        model_output_dir=args.model_output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit, # Passando o argumento
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Passando o argumento
    )
