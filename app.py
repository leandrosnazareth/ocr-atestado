import streamlit as st
import os
import json
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import subprocess # Para rodar o script de fine-tuning
import re # Para parsing básico dos campos

# Diretório onde o modelo e o processador serão salvos/carregados
MODEL_DIR = "./fine_tuned_trocr_model"
DATA_UPLOAD_DIR = "./uploaded_data" # Para salvar os JSONs de treinamento
TEMP_IMAGES_DIR = "./temp_images" # Diretório temporário para salvar imagens para inferência

# Garante que os diretórios existam ao iniciar a aplicação
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

# Função para carregar o modelo e o processador para inferência
@st.cache_resource # Cacheia o modelo e o processador para não recarregar a cada interação
def load_model_for_inference():
    """
    Carrega o processador e o modelo TrOCR.
    Tenta carregar o modelo fine-tuned primeiro, se não existir, carrega o modelo base.
    """
    processor = None
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR): # Verifica se o diretório existe e não está vazio
        try:
            st.info(f"Carregando modelo fine-tuned de: {MODEL_DIR}...")
            processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
            if device == "cuda":
                model.to("cuda")
            st.success("Modelo fine-tuned carregado com sucesso!")
        except Exception as e:
            st.warning(f"Erro ao carregar o modelo fine-tuned ({e}). Carregando modelo base...")
            # Fallback para o modelo base se o fine-tuned não carregar
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            if device == "cuda":
                model.to("cuda")
            st.info("Modelo base 'microsoft/trocr-base-handwritten' carregado.")
    else:
        st.warning("Diretório do modelo fine-tuned não encontrado ou vazio. Carregando modelo base 'microsoft/trocr-base-handwritten'...")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        if device == "cuda":
            model.to("cuda")
        st.info("Modelo base 'microsoft/trocr-base-handwritten' carregado.")

    return processor, model

# Função para realizar inferência (OCR) em uma imagem
def perform_ocr(image_path, processor, model):
    """
    Executa o OCR em uma imagem usando o modelo TrOCR.
    """
    if processor is None or model is None:
        return "Erro: Modelo ou processador não carregado. Verifique os logs."
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Move os pixel_values para a GPU se o modelo estiver na GPU
        if torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        generated_ids = model.generate(pixel_values)
        generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    except Exception as e:
        return f"Erro durante a inferência: {e}"

# Função para salvar o arquivo de dados JSONL de treinamento
def save_uploaded_training_data(uploaded_file, save_dir):
    """
    Salva um arquivo carregado pelo usuário no diretório especificado.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Função para parsing de campos específicos do texto extraído
def parse_medical_certificate_text(text):
    """
    Extrai campos específicos do texto OCR usando expressões regulares.
    Esta é uma implementação básica e pode precisar de ajustes
    dependendo da variabilidade do formato dos seus atestados.
    """
    parsed_data = {}

    # Expressões regulares para cada campo
    # O objetivo é capturar o valor após a chave e antes do próximo delimitador (;) ou fim da string
    patterns = {
        "nome": r"(?:nome|paciente)[:\s]*([^;,\n]+)",
        "data": r"(?:data|em)[:\s]*(\d{2}[/\-]\d{2}[/\-]\d{4})", # Formato DD/MM/AAAA ou DD-MM-AAAA
        "cid": r"(?:cid|c\.i\.d)[:\s]*([A-Z]\d{2}(?:\.\d{1,2})?)", # Ex: A01.0, J45
        "dias_afastamento": r"(?:dias de afastamento|dias)[:\s]*(\d+)",
        "crm": r"(?:crm|c\.r\.m)[:\s]*(\d+(?:[A-Z]{2})?)", # Ex: 123456SP, 12345
        "medico": r"(?:médico|dr\.?|dra\.?)[:\s]*([^;,\n]+)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Limpa o valor capturado, removendo espaços extras
            value = match.group(1).strip()
            # Heurística para evitar capturar o próximo campo como parte do valor
            if ';' in value:
                value = value.split(';')[0].strip()
            parsed_data[key] = value

    return parsed_data

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="TrOCR para Atestados Médicos")

st.title("TrOCR para Atestados Médicos")
st.markdown("Uma aplicação para fine-tuning e inferência de modelos TrOCR em atestados médicos.")

# Barra lateral para navegação
st.sidebar.title("Navegação")
# Atualização dos nomes das opções para maior clareza e consistência
option = st.sidebar.radio("Selecione uma opção:", ("Treinamento do Modelo", "Inferência de Atestado"))

# --- Seção de Treinamento ---
if option == "Treinamento do Modelo":
    st.header("Treinamento do Modelo TrOCR")
    st.write("Faça upload do seu arquivo JSONL com os dados de treinamento para fine-tuning do TrOCR.")
    st.info("O arquivo JSONL deve ter uma linha por objeto JSON, no formato: "
            "`{\"image\": \"caminho/para/imagem.jpg\", \"target_text\": \"nome: ...; data: ...; cid: ...; dias_afastamento: ...; crm: ...\"}`. "
            "Os caminhos das imagens devem ser relativos ao diretório onde o Docker container será executado (ex: `./images/atestado1.jpg`). "
            "Certifique-se de que o diretório `images` esteja montado no container.")

    uploaded_training_file = st.file_uploader("Escolha um arquivo JSONL para treinamento", type=["jsonl"])

    if uploaded_training_file is not None:
        st.success("Arquivo JSONL carregado com sucesso!")
        training_data_path = save_uploaded_training_data(uploaded_training_file, DATA_UPLOAD_DIR)
        st.write(f"Arquivo salvo para treinamento em: `{training_data_path}`")

        st.subheader("Configurações de Treinamento")
        epochs = st.slider("Épocas de Treinamento", min_value=1, max_value=20, value=3, help="Número de vezes que o modelo verá todo o dataset de treinamento.")
        batch_size = st.slider("Tamanho do Batch", min_value=1, max_value=16, value=2, help="Número de amostras processadas antes de atualizar os pesos do modelo.")
        learning_rate = st.number_input("Taxa de Aprendizado", min_value=1e-7, max_value=1e-3, value=4e-5, format="%e", help="Determina o tamanho do passo em que os pesos do modelo são ajustados.")

        if st.button("Iniciar Treinamento", type="primary"):
            st.info("Iniciando treinamento... Isso pode levar algum tempo, dependendo do tamanho do dataset e do hardware.")
            st.warning("A saída do treinamento será exibida abaixo. O Streamlit pode parecer congelado durante o processo.")

            try:
                # CORREÇÃO CRÍTICA: Mudar "python" para "python3" para garantir que o interpretador correto seja encontrado no Docker
                command = [
                    "python3", "fine_tuning_script.py",
                    "--data_json_path", training_data_path,
                    "--model_output_dir", MODEL_DIR,
                    "--epochs", str(epochs),
                    "--batch_size", str(batch_size),
                    "--learning_rate", str(learning_rate)
                ]
                # Usa st.empty() para criar um placeholder para a saída do processo
                output_area = st.empty()
                output_area.code("Aguardando saída do treinamento...")

                # Executa o subprocesso e captura a saída em tempo real
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                stdout_lines = []
                stderr_lines = []

                # Lê a saída em tempo real
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        stdout_lines.append(output)
                        output_area.code("".join(stdout_lines)) # Atualiza o placeholder com a nova saída

                # Captura qualquer erro restante
                stderr = process.stderr.read()
                if stderr:
                    stderr_lines.append(stderr)

                process.wait() # Espera o processo terminar completamente

                if process.returncode == 0:
                    st.success("Treinamento concluído com sucesso!")
                    st.text("Saída Completa do Treinamento:")
                    st.code("".join(stdout_lines))
                else:
                    st.error("Erro durante o treinamento!")
                    st.text("Saída de Erro:")
                    st.code("".join(stderr_lines))
                    st.text("Saída Padrão (se houver):")
                    st.code("".join(stdout_lines))

            except Exception as e:
                st.error(f"Erro ao tentar iniciar o treinamento: {e}")
    else:
        st.warning("Por favor, faça upload do seu arquivo JSONL para iniciar o treinamento.")


# --- Seção de Inferência ---
elif option == "Inferência de Atestado": # Nome da opção atualizado
    st.header("Inferência de Atestados Médicos")
    st.write("Faça upload de um atestado médico (JPG) para extrair o texto e os campos específicos.")

    # Carrega o processador e o modelo (fine-tuned ou base).
    processor, model = load_model_for_inference()

    # Widget para upload da imagem
    uploaded_image = st.file_uploader("Escolha um arquivo JPG", type=["jpg", "jpeg"])

    if uploaded_image is not None:
        # Divide a tela em duas colunas para melhor visualização: imagem à esquerda, resultados à direita.
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_image, caption="Atestado Carregado", use_column_width=True)

        with col2:
            st.info("Realizando OCR e extração de campos...")

            # Bloco try-except-finally para garantir a limpeza do arquivo temporário e lidar com erros.
            try:
                # Salva a imagem temporariamente para o modelo poder lê-la.
                # O diretório TEMP_IMAGES_DIR já é criado no início do script.
                image_path = os.path.join(TEMP_IMAGES_DIR, uploaded_image.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                # Realiza o OCR utilizando a função perform_ocr.
                extracted_text = perform_ocr(image_path, processor, model)
                st.subheader("Texto Extraído:")
                st.success(extracted_text)

                # Extrai os campos utilizando a função de parsing baseada em expressões regulares.
                st.subheader("Campos Extraídos:")
                parsed_fields = parse_medical_certificate_text(extracted_text)

                if parsed_fields:
                    # Exibe os campos extraídos em formato JSON para fácil visualização estruturada.
                    st.json(parsed_fields)
                    
                    # Opcional: Se preferir uma exibição mais formatada, descomente o bloco abaixo:
                    # st.markdown("---") # Separador visual
                    # for key, value in parsed_fields.items():
                    #     # Formata a chave para ser mais legível (ex: "dias_afastamento" -> "Dias Afastamento")
                    #     st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.warning("Não foi possível extrair campos específicos. Verifique o formato do texto ou ajuste a lógica de parsing.")

            except Exception as e:
                # Exibe uma mensagem de erro genérica em caso de falha no processamento.
                st.error(f"Ocorreu um erro durante o processamento: {e}")
            finally:
                # Garante que o arquivo temporário seja removido após o uso, independentemente de sucesso ou erro.
                if os.path.exists(image_path):
                    os.remove(image_path)
    else:
        st.info("Carregue uma imagem JPG para iniciar a inferência.")

