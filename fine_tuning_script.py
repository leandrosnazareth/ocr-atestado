import os
import json
import argparse
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# --- 1. Definição do Dataset Personalizado ---
class MedicalCertificateDataset(Dataset):
    """
    Dataset personalizado para carregar imagens de atestados médicos e seus textos alvo.
    """
    def __init__(self, data_path, processor, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        """
        Carrega os dados de um arquivo JSONL (JSON Lines).
        Cada linha do arquivo deve ser um objeto JSON.
        """
        data = []
        if not os.path.exists(data_path):
            print(f"Erro: O arquivo de dados não foi encontrado em {data_path}")
            return []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON no arquivo {data_path}: {e}")
            print(f"Linha que causou o erro: {line.strip()}")
            return []
        return data

    def __len__(self):
        """Retorna o número total de itens no dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retorna um item do dataset (imagem processada e labels de texto).
        """
        item = self.data[idx]
        image_path = item["image"]
        target_text = item["target_text"]

        # Verifica se o caminho da imagem é absoluto ou relativo
        # Se for relativo, assume que é relativo ao diretório de trabalho atual
        if not os.path.isabs(image_path):
            # Assumimos que as imagens estão em um diretório 'images' ou no mesmo nível
            # Para o Docker, certifique-se de que o diretório de imagens seja montado corretamente.
            # Ex: se o caminho no JSON for 'images/atestado1.jpg' e 'images' está na raiz do projeto
            # ele será encontrado.
            pass # O caminho já deve estar correto se o volume Docker for montado apropriadamente

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Aviso: Imagem não encontrada em {image_path}. Pulando este item.")
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
        # Isso garante que a perda não seja calculada sobre os tokens de padding.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values.squeeze(), "labels": labels.squeeze()}

# Função para filtrar itens None do DataLoader
def collate_fn(batch):
    """
    Função de colagem para o DataLoader, que filtra itens None (imagens não encontradas ou com erro).
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Retorna None se o batch estiver vazio após a filtragem

    # Empilha os tensores de pixel_values e labels
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"pixel_values": pixel_values, "labels": labels}

# --- 2. Função Principal de Fine-tuning ---
def run_fine_tuning(
    data_json_path,
    model_output_dir="fine_tuned_trocr_model",
    epochs=10,
    batch_size=8,
    learning_rate=5e-5,
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
):
    """
    Executa o processo de fine-tuning do modelo TrOCR.

    Args:
        data_json_path (str): Caminho para o arquivo JSONL contendo os dados de treinamento.
        model_output_dir (str): Diretório para salvar o modelo e o processador treinados.
        epochs (int): Número de épocas de treinamento.
        batch_size (int): Tamanho do batch para treinamento e avaliação.
        learning_rate (float): Taxa de aprendizado para o otimizador.
        eval_steps (int): Número de passos entre cada avaliação.
        save_steps (int): Número de passos entre cada salvamento do modelo.
        logging_steps (int): Número de passos entre cada log de treinamento.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Carregar processador e modelo pré-treinado da Microsoft
    print("Carregando processador e modelo pré-treinado 'microsoft/trocr-base-handwritten'...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    # Configurar o decodificador para geração de texto
    # Estes são parâmetros importantes para a geração de texto do modelo.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Criar dataset personalizado
    print(f"Carregando dados do dataset de: {data_json_path}")
    dataset = MedicalCertificateDataset(data_json_path, processor)

    if not dataset:
        print("Nenhum dado válido encontrado no dataset. Abortando treinamento.")
        return

    # Dividir o dataset em treino e validação (90% treino, 10% validação)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Garante que a divisão seja aleatória e reproduzível
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    print(f"Tamanho do dataset de treino: {len(train_dataset)}")
    print(f"Tamanho do dataset de avaliação: {len(eval_dataset)}")

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
        logging_steps=logging_steps, # Loga métricas a cada 'logging_steps' passos
        logging_dir="./logs", # Diretório para logs (ex: para TensorBoard)
        report_to="none", # Desabilita integração com Weights & Biases, etc.
        load_best_model_at_end=True, # Carrega o melhor modelo (baseado em eval_loss) no final do treinamento
        metric_for_best_model="eval_loss",
        greater_is_better=False, # Para loss, menor é melhor
        fp16=True if device == "cuda" else False, # Habilita Mixed Precision Training para GPU (mais rápido, menos memória)
        dataloader_num_workers=os.cpu_count() // 2 if os.cpu_count() else 0, # Otimiza o carregamento de dados
    )

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn, # Usa a função de colagem personalizada
    )

    # Treinar o modelo
    print("Iniciando treinamento...")
    trainer.train()
    print("Treinamento concluído!")

    # Salvar o modelo e o processador treinados
    print(f"Salvando modelo e processador em {model_output_dir}...")
    os.makedirs(model_output_dir, exist_ok=True) # Garante que o diretório exista
    processor.save_pretrained(model_output_dir)
    model.save_pretrained(model_output_dir)
    print("Modelo e processador salvos.")

    # --- Exemplo de Inferência (Após o treinamento) ---
    print("\n--- Exemplo de Inferência após o treinamento ---")
    if eval_dataset and len(eval_dataset) > 0:
        try:
            # Pega um item do dataset de validação para teste
            sample_item = eval_dataset[0]
            # Adiciona dimensão de batch e move para o dispositivo correto
            pixel_values_sample = sample_item["pixel_values"].unsqueeze(0).to(device)

            # Gera o texto a partir da imagem
            generated_ids = model.generate(pixel_values_sample)
            generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Decodifica o texto original (label) para comparação
            original_text = processor.tokenizer.decode(
                sample_item['labels'][sample_item['labels'] != -100],
                skip_special_tokens=True
            )
            print(f"Texto Original: {original_text}")
            print(f"Texto Gerado: {generated_text}")
        except IndexError:
            print("Nenhum dado de avaliação disponível para exemplo de inferência.")
        except Exception as e:
            print(f"Erro durante o exemplo de inferência: {e}")
    else:
        print("Nenhum dataset de avaliação disponível para exemplo de inferência.")


if __name__ == "__main__":
    # Configuração do parser de argumentos para permitir que o Streamlit passe parâmetros
    parser = argparse.ArgumentParser(description="Script de Fine-tuning do TrOCR para Atestados Médicos.")
    parser.add_argument("--data_json_path", type=str, required=True,
                        help="Caminho para o arquivo JSONL contendo os dados de treinamento.")
    parser.add_argument("--model_output_dir", type=str, default="fine_tuned_trocr_model",
                        help="Diretório para salvar o modelo e o processador treinados.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Número de épocas de treinamento.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Tamanho do batch para treinamento e avaliação.")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Taxa de aprendizado para o otimizador.")

    args = parser.parse_args()

    # Rodar o fine-tuning com os argumentos fornecidos
    run_fine_tuning(
        data_json_path=args.data_json_path,
        model_output_dir=args.model_output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

