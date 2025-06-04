# Use uma imagem base Python com CUDA, se você planeja usar GPU
# Se não tiver GPU ou não quiser usar, substitua por "python:3.9-slim-buster"
# A versão do CUDA deve ser compatível com sua GPU e a versão do PyTorch.
# Esta é uma boa opção para CUDA 11.8:
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Define a versão do Python para instalar
ENV PYTHON_VERSION=3.9
ENV DEBIAN_FRONTEND=noninteractive

# Instala Python e dependências essenciais do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3-venv \
    python3-pip \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# REMOVIDO: A linha 'update-alternatives' foi removida para evitar o erro.
# Em vez disso, usaremos 'python3' explicitamente para pip e streamlit,
# pois 'python3' geralmente aponta para a versão mais recente instalada.

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de requisitos para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências Python
# Instala PyTorch com suporte a CUDA 11.8
# Garante que a versão do PyTorch seja compatível com o CUDA 11.8 da imagem base.
# Usa 'python3 -m pip' para garantir que pip esteja associado ao interpretador python3
RUN python3 -m pip install --no-cache-dir torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Copia o restante da aplicação para o diretório de trabalho
COPY . .

# Cria diretórios para dados de upload, modelo e imagens temporárias
RUN mkdir -p ./uploaded_data ./temp_images ./fine_tuned_trocr_model ./logs

# Expõe a porta que o Streamlit usará
EXPOSE 8502

# Comando para iniciar a aplicação Streamlit
# Usa 'python3 -m streamlit' para garantir que o streamlit seja executado com o interpretador python3
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]


