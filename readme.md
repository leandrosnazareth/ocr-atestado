# 🏥 Sistema de Extração de Dados de Atestados Médicos com TrOCR

Este projeto implementa uma solução para extração automática de campos específicos de atestados médicos em formato JPG utilizando o modelo **TrOCR** (Transformer-based Optical Character Recognition) da Microsoft.

A aplicação permite realizar o *fine-tuning* do modelo com uma base de dados personalizada e oferece uma interface gráfica intuitiva para upload de arquivos e controle do processo de treinamento e inferência. Todo o ambiente é empacotado em um container Docker para facilitar a implantação e garantir a reprodutibilidade.

---

## 🔍 Visão Geral do Projeto

O core do projeto consiste em:

- 🎯 **Fine-tuning do TrOCR**  
  Adaptação de um modelo TrOCR pré-treinado para o domínio específico de atestados médicos, permitindo um reconhecimento de texto otimizado.

- 🔎 **Extração de Campos**  
  Após o OCR, uma etapa de pós-processamento com expressões regulares extrai campos-chave como:
  - 👤 Nome
  - 📅 Data
  - 🏷️ CID
  - ⏳ Dias de afastamento
  - 🩺 CRM do médico

- 💻 **Interface Gráfica (Streamlit)**  
  Aplicação web simples e funcional que facilita:
  - **Treinamento**: Upload de dados e configuração de hiperparâmetros.
  - **Inferência**: Upload de novos atestados para extração automática de texto e campos.

- 🐳 **Containerização (Docker)**  
  Todo o ambiente da aplicação, incluindo dependências e modelo, é empacotado em uma imagem Docker, garantindo execução consistente e reprodutível.

---

## 📁 Estrutura do Projeto

```plaintext
seu_projeto/
├── fine_tuning_script.py
├── app.py
├── requirements.txt
├── Dockerfile
├── images/
│   ├── atestado1.jpg
│   └── atestado2.jpg
└── data.jsonl
```


- **`fine_tuning_script.py`**: Contém a lógica para carregar o dataset personalizado de atestados, configurar o modelo TrOCR e executar o processo de fine-tuning.
- **`app.py`**: A aplicação Streamlit que fornece a interface gráfica para interação com o usuário (upload de dados, início do treinamento, inferência).
- **`requirements.txt`**: Lista todas as bibliotecas Python necessárias para o projeto (Hugging Face Transformers, PyTorch, Streamlit, etc.).
- **`Dockerfile`**: Define as etapas para construir a imagem Docker do seu projeto, instalando as dependências e configurando o ambiente.
- **`images/`** (Opcional, mas recomendado): Um diretório onde você deve armazenar suas imagens de atestados médicos para treinamento. Os caminhos no seu `data.jsonl` devem apontar para estas imagens.
- **`data.jsonl`**: Seu arquivo de dados de treinamento. Cada linha deve ser um objeto JSON no formato:

```json
{
    "image": "./images/atestado_exemplo.jpg", 
    "target_text": "nome: Nome do Paciente; data: DD/MM/AAAA; cid: CXX.X; dias_afastamento: X; crm: 123456"
}
```


## ⚙️ Requisitos

- **Docker**: Certifique-se de ter o Docker Desktop (Windows/macOS) ou o Docker Engine (Linux) instalado e em execução.
- **GPU (Opcional, mas Altamente Recomendado para Treinamento)**: Se você pretende realizar o fine-tuning usando uma GPU NVIDIA, certifique-se de ter os drivers NVIDIA, CUDA Toolkit e NVIDIA Container Toolkit instalados em seu sistema host, compatíveis com a versão do CUDA especificada no Dockerfile (atualmente 11.8). Sem GPU, o treinamento será significativamente mais lento.

## 🚀 Instruções de Implantação

Siga os passos abaixo para implantar e executar o projeto:

### 1. 🛠️ Preparação do Ambiente

    ```bash
    # Clone o repositório
    git https://github.com/leandrosnazareth/ocr-atestado.git
    cd ocr-atestado/
    ```


### 2. 🏗️ Construção da Imagem Docker

- **Construa a Imagem Docker**:
    Execute o comando abaixo para construir a imagem Docker. Isso pode levar alguns minutos na primeira vez, pois o Docker precisa baixar a imagem base e instalar todas as dependências.

    ```bash
    docker build -t trocr-atestados .
    ```

    **Para usuários sem GPU**: Se você não tem uma GPU NVIDIA ou não deseja utilizá-la, edite o Dockerfile e mude a linha:

    ```dockerfile
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
    ```

    Para:

    ```dockerfile
    FROM python:3.9-slim-buster
    ```

    Além disso, remova a linha de instalação do PyTorch com CUDA:

    ```bash
    pip install --no-cache-dir torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
    ```

    Após essa alteração, construa a imagem com o mesmo comando:

    ```bash
    docker build -t trocr-atestados .
    ```

## 3. 🏃 Execução do Container Docker

- **Execute o Container**: Execute o comando apropriado abaixo para iniciar o container Docker. Os volumes (`-v`) são cruciais para persistir seus dados e o modelo treinado, além de permitir que o container acesse suas imagens de treinamento.

### Com GPU (Recomendado para Treinamento)

```bash
docker run --gpus all -p 8501:8501 \
-v "$(pwd)/fine_tuned_trocr_model:/app/fine_tuned_trocr_model" \
-v "$(pwd)/uploaded_data:/app/uploaded_data" \
-v "$(pwd)/temp_images:/app/temp_images" \
-v "$(pwd)/images:/app/images" \
-v "$(pwd)/logs:/app/logs" \
trocr-atestados
```

### Sem GPU (após ajustar o Dockerfile):
```bash
docker run -p 8501:8501 \
-v "$(pwd)/fine_tuned_trocr_model:/app/fine_tuned_trocr_model" \
-v "$(pwd)/uploaded_data:/app/uploaded_data" \
-v "$(pwd)/temp_images:/app/temp_images" \
-v "$(pwd)/images:/app/images" \
-v "$(pwd)/logs:/app/logs" \
trocr-atestados
```

## 4. Acesso à Aplicação

- **Abra seu navegador**: Após o container iniciar (pode levar alguns segundos), acesse:

  [http://localhost:8501](http://localhost:8501)

## 💡 Utilizando a Aplicação

Uma vez que a aplicação Streamlit esteja carregada no seu navegador:

### 🏋️ Treinamento do Modelo:
1. Navegue para a aba **"Treinamento do Modelo"**.
2. Faça upload do seu arquivo `data.jsonl`.
3. Ajuste os hiperparâmetros (Épocas, Tamanho do Batch, Taxa de Aprendizado) conforme necessário.
4. Clique em **"Iniciar Treinamento"**. A saída do treinamento será exibida na interface. Aguarde a conclusão para que o modelo seja salvo.

### 🔮 Inferência de Atestado:
1. Navegue para a aba **"Inferência de Atestado"**.
2. Faça upload de um arquivo JPG de atestado médico.
3. A aplicação executará o OCR e tentará extrair os campos definidos (nome, data, CID, etc.). O texto completo e os campos extraídos serão exibidos.