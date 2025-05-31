import streamlit as st
import os
import json
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import subprocess # Para rodar o script de fine-tuning

# Diretório onde o modelo e o processador serão salvos/carregados
MODEL_DIR = "./fine_tuned_trocr_model"
DATA_UPLOAD_DIR = "./uploaded_data" # Para salvar os JSONs de treinamento

# Função para realizar inferência
@st.cache_resource # Cacheia o modelo e o processador para não recarregar a cada interação
def load_model_for_inference():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Modelo não encontrado em: {MODEL_DIR}. Por favor, treine o modelo primeiro.")
        return None, None
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
        # Move o modelo para a GPU se disponível
        if torch.cuda.is_available():
            model.to("cuda")
        return processor, model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo ou processador: {e}")
        return None, None

def perform_ocr(image_path, processor, model):
    if processor is None or model is None:
        return "Erro: Modelo ou processador não carregado."
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

# Função para salvar o arquivo de dados JSON de treinamento
def save_uploaded_training_data(uploaded_file, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- Interface Streamlit ---
st.title("TrOCR para Atestados Médicos")
st.write("Faça fine-tuning do modelo TrOCR e realize inferência em atestados médicos.")

# Barra lateral para navegação
st.sidebar.title("Navegação")
option = st.sidebar.radio("Selecione uma opção:", ("Treinamento", "Inferência"))

# --- Seção de Treinamento ---
if option == "Treinamento":
    st.header("Treinamento do Modelo TrOCR")
    st.write("Faça upload do seu arquivo JSONL com os dados de treinamento.")
    st.info("O arquivo JSONL deve ter uma linha por objeto JSON, no formato: "
            "`{\"image\": \"caminho/para/imagem.jpg\", \"target_text\": \"nome: ...; data: ...; cid: ...; dias_afastamento: ...; crm: ...\"}`. "
            "Certifique-se de que os caminhos das imagens sejam relativos ao local onde o Docker container será executado ou sejam caminhos absolutos acessíveis.")

    uploaded_training_file = st.file_uploader("Escolha um arquivo JSONL para treinamento", type="jsonl")

    if uploaded_training_file is not None:
        st.success("Arquivo JSONL carregado com sucesso!")
        training_data_path = save_uploaded_training_data(uploaded_training_file, DATA_UPLOAD_DIR)
        st.write(f"Arquivo salvo em: {training_data_path}")

        st.subheader("Configurações de Treinamento")
        epochs = st.slider("Épocas de Treinamento", min_value=1, max_value=20, value=3)
        batch_size = st.slider("Tamanho do Batch", min_value=1, max_value=16, value=2)
        learning_rate = st.number_input("Taxa de Aprendizado", min_value=1e-6, max_value=1e-3, value=4e-5, format="%e")

        if st.button("Iniciar Treinamento"):
            st.info("Iniciando treinamento... Isso pode levar algum tempo.")
            try:
                # Chama o script de fine-tuning em um processo separado
                # Passa os argumentos via linha de comando
                command = [
                    "python", "fine_tuning_script.py",
                    "--data_json_path", training_data_path,
                    "--model_output_dir", MODEL_DIR,
                    "--epochs", str(epochs),
                    "--batch_size", str(batch_size),
                    "--learning_rate", str(learning_rate)
                ]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate() # Espera o processo terminar

                if process.returncode == 0:
                    st.success("Treinamento concluído com sucesso!")
                    st.text("Saída do Treinamento:")
                    st.code(stdout)
                else:
                    st.error("Erro durante o treinamento!")
                    st.code(stderr)
            except Exception as e:
                st.error(f"Erro ao tentar iniciar o treinamento: {e}")
    else:
        st.warning("Por favor, faça upload do seu arquivo JSONL para iniciar o treinamento.")


# --- Seção de Inferência ---
elif option == "Inferência":
    st.header("Inferência de Atestados Médicos")
    st.write("Faça upload de um atestado médico (JPG) para extrair o texto.")

    processor, model = load_model_for_inference()

    uploaded_image = st.file_uploader("Escolha um arquivo JPG", type=["jpg", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Atestado Carregado", use_column_width=True)
        st.write("")
        st.info("Realizando OCR...")

        # Salva a imagem temporariamente para o modelo poder lê-la
        image_path = os.path.join("./temp_images", uploaded_image.name)
        os.makedirs("./temp_images", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        extracted_text = perform_ocr(image_path, processor, model)
        st.subheader("Texto Extraído:")
        st.success(extracted_text)

        # Opcional: Aqui você pode adicionar a lógica de parsing para extrair campos específicos
        # usando regex ou um modelo de PLN, como discutido.
        st.subheader("Extração de Campos (Exemplo - Necessita lógica de parsing):")
        # Exemplo muito básico de parsing (você precisará de algo mais robusto)
        if "nome:" in extracted_text and "data:" in extracted_text:
            try:
                nome = extracted_text.split("nome:")[1].split(";")[0].strip()
                data = extracted_text.split("data:")[1].split(";")[0].strip()
                cid = extracted_text.split("cid:")[1].split(";")[0].strip()
                dias_afastamento = extracted_text.split("dias_afastamento:")[1].split(";")[0].strip()
                crm = extracted_text.split("crm:")[1].split(";")[0].strip()
                # Exibe os campos extraídos
                st.write(f"**Nome:** {nome}")
                st.write(f"**Data:** {data}")
                st.write(f"**CID:** {cid}")
                st.write(f"**Dias de Afastamento:** {dias_afastamento}")
                st.write(f"**CRM:** {crm}")
                
            except IndexError:
                st.warning("Não foi possível extrair todos os campos com o parsing atual. Ajuste a lógica.")
        else:
            st.warning("Lógica de parsing de campos ainda não aplicada ou texto extraído incompleto.")

        # Limpa o arquivo temporário
        os.remove(image_path)
