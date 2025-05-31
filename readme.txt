README: Sistema de Extração de Dados de Atestados Médicos com TrOCR

Este projeto implementa uma solução para extração automática de campos específicos de atestados médicos em formato JPG utilizando o modelo TrOCR (Transformer-based Optical Character Recognition) da Microsoft. A aplicação permite realizar o fine-tuning do modelo com uma base de dados personalizada e oferece uma interface gráfica intuitiva para upload de arquivos e controle do processo de treinamento e inferência. Todo o ambiente é empacotado em um container Docker para facilitar a implantação e garantir a reprodutibilidade.
Visão Geral do Projeto

O core do projeto consiste em:

    Fine-tuning do TrOCR: Adaptação de um modelo TrOCR pré-treinado para o domínio específico de atestados médicos, permitindo um reconhecimento de texto otimizado para este tipo de documento.
    Extração de Campos: Após o reconhecimento do texto (OCR), uma etapa de pós-processamento utiliza expressões regulares para extrair campos-chave como nome, data, CID, dias de afastamento, e CRM do médico.
    Interface Gráfica (Streamlit): Uma aplicação web simples e funcional construída com Streamlit que facilita:
        Treinamento: Upload de bases de dados de atestados anotados e configuração de hiperparâmetros para o fine-tuning.
        Inferência: Upload de novos atestados médicos para extração automática de texto e campos.
    Containerização (Docker): Todo o ambiente da aplicação, incluindo dependências e o modelo, é empacotado em uma imagem Docker. Isso garante que o projeto possa ser implantado e executado de forma consistente em qualquer ambiente que suporte Docker, eliminando problemas de compatibilidade de dependências.

Estrutura do Projeto

A estrutura de diretórios do projeto deve ser a seguinte:

seu_projeto/
├── fine_tuning_script.py
├── app.py
├── requirements.txt
├── Dockerfile
├── images/
│   ├── atestado1.jpg
│   └── atestado2.jpg
└── data.jsonl

    fine_tuning_script.py: Contém a lógica para carregar o dataset personalizado de atestados, configurar o modelo TrOCR e executar o processo de fine-tuning.
    app.py: A aplicação Streamlit que fornece a interface gráfica para interação com o usuário (upload de dados, início do treinamento, inferência).
    requirements.txt: Lista todas as bibliotecas Python necessárias para o projeto (Hugging Face Transformers, PyTorch, Streamlit, etc.).
    Dockerfile: Define as etapas para construir a imagem Docker do seu projeto, instalando as dependências e configurando o ambiente.
    images/: (Opcional, mas recomendado) Um diretório onde você deve armazenar suas imagens de atestados médicos para treinamento. Os caminhos no seu data.jsonl devem apontar para estas imagens.
    data.jsonl: Seu arquivo de dados de treinamento. Cada linha deve ser um objeto JSON no formato:
    JSON

    {"image": "./images/atestado_exemplo.jpg", "target_text": "nome: Nome do Paciente; data: DD/MM/AAAA; cid: CXX.X; dias_afastamento: X; crm: 123456"}

    Importante: Os caminhos das imagens ("./images/atestado_exemplo.jpg") no data.jsonl devem ser relativos ao diretório raiz do projeto (/app dentro do container Docker) e corresponder à estrutura de pastas que você montar no container.

Requisitos

    Docker: Certifique-se de ter o Docker Desktop (Windows/macOS) ou o Docker Engine (Linux) instalado e em execução.
    GPU (Opcional, mas Altamente Recomendado para Treinamento): Se você pretende realizar o fine-tuning usando uma GPU NVIDIA, certifique-se de ter os drivers NVIDIA, CUDA Toolkit e NVIDIA Container Toolkit instalados em seu sistema host, compatíveis com a versão do CUDA especificada no Dockerfile (atualmente 11.8). Sem GPU, o treinamento será significativamente mais lento.

Instruções de Implantação

Siga os passos abaixo para implantar e executar o projeto:
1. Preparação do Ambiente

    Clone ou Baixe o Projeto: Obtenha todos os arquivos do projeto (fine_tuning_script.py, app.py, requirements.txt, Dockerfile) e coloque-os em um diretório no seu computador (ex: ocr-atestado/).

    Crie a Pasta de Imagens e o Arquivo de Dados:
        Dentro do diretório raiz do seu projeto (ex: ocr-atestado/), crie uma pasta chamada images. Coloque todas as suas imagens JPG de atestados médicos que serão usadas para treinamento dentro desta pasta.
        Crie o arquivo data.jsonl na raiz do seu projeto. Popule-o com suas anotações no formato JSONL (um objeto JSON por linha), garantindo que os caminhos das imagens no data.jsonl correspondam aos arquivos na pasta images/ (ex: "image": "./images/nome_da_imagem.jpg").

2. Construção da Imagem Docker

    Abra o Terminal: Navegue até o diretório raiz do seu projeto (ocr-atestado/) no terminal.

    Construa a Imagem Docker:
    Execute o comando a seguir para construir a imagem Docker. Isso pode levar alguns minutos na primeira vez, pois o Docker precisa baixar a imagem base e instalar todas as dependências.
    Bash

    docker build -t trocr-atestados .

        Para usuários sem GPU: Se você não tem uma GPU NVIDIA ou não deseja utilizá-la, você deve editar o Dockerfile e mudar a linha FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 para FROM python:3.9-slim-buster. Além disso, remova a linha de instalação do PyTorch com CUDA (ou seja, pip install --no-cache-dir torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \). Após essa alteração, construa a imagem com o mesmo comando docker build -t trocr-atestados ..

3. Execução do Container Docker

    Execute o Container:
    Execute o comando apropriado abaixo para iniciar o container Docker. Os volumes (-v) são cruciais para persistir seus dados e o modelo treinado, e para que o container possa acessar suas imagens de treinamento.

    Com GPU (Recomendado para Treinamento):

    docker run --gpus all -p 8501:8501 \
    -v "$(pwd)/fine_tuned_trocr_model:/app/fine_tuned_trocr_model" \
    -v "$(pwd)/uploaded_data:/app/uploaded_data" \
    -v "$(pwd)/temp_images:/app/temp_images" \
    -v "$(pwd)/images:/app/images" \
    -v "$(pwd)/logs:/app/logs" \
    trocr-atestados

    --gpus all: Permite que o container acesse todas as GPUs disponíveis no seu sistema.
    -p 8501:8501: Mapeia a porta 8501 do container para a porta 8501 do seu host, permitindo que você acesse a aplicação.
    -v "$(pwd)/<HOST_DIR>:/app/<CONTAINER_DIR>": Monta diretórios do seu host ($(pwd) aponta para o diretório atual) dentro do container. Isso garante que:
        fine_tuned_trocr_model: O modelo treinado será salvo e carregado persistindo entre as execuções do container.
        uploaded_data: Seus arquivos JSONL de treinamento carregados pela interface serão salvos aqui.
        temp_images: Imagens temporárias para inferência serão gerenciadas aqui.
        images: Suas imagens de treinamento (no diretório images/ do seu host) serão acessíveis ao script de treinamento dentro do container. Certifique-se de que este caminho esteja correto.
        logs: Logs de treinamento serão salvos aqui.

    Sem GPU (após ajustar o Dockerfile):

        docker run -p 8501:8501 \
        -v "$(pwd)/fine_tuned_trocr_model:/app/fine_tuned_trocr_model" \
        -v "$(pwd)/uploaded_data:/app/uploaded_data" \
        -v "$(pwd)/temp_images:/app/temp_images" \
        -v "$(pwd)/images:/app/images" \
        -v "$(pwd)/logs:/app/logs" \
        trocr-atestados

4. Acesso à Aplicação

    Abra seu navegador: Após o container iniciar (pode levar alguns segundos), acesse:

    http://localhost:8501

Utilizando a Aplicação

Uma vez que a aplicação Streamlit esteja carregada no seu navegador:

    Treinamento do Modelo:
        Navegue para a aba "Treinamento do Modelo".
        Faça upload do seu arquivo data.jsonl.
        Ajuste os hiperparâmetros (Épocas, Tamanho do Batch, Taxa de Aprendizado) conforme necessário.
        Clique em "Iniciar Treinamento". A saída do treinamento será exibida na interface. Aguarde a conclusão para que o modelo seja salvo.

    Inferência de Atestado:
        Navegue para a aba "Inferência de Atestado".
        Faça upload de um arquivo JPG de atestado médico.
        A aplicação executará o OCR e tentará extrair os campos definidos (nome, data, CID, etc.). O texto completo e os campos extraídos serão exibidos.

Considerações Importantes e Próximos Passos

    Qualidade dos Dados de Treinamento: A performance do modelo fine-tuned depende diretamente da qualidade e quantidade dos seus dados anotados em data.jsonl. Quanto mais diversificados e precisos forem seus atestados e anotações, melhor o modelo se sairá.
    Parsing de Campos: A extração de campos específicos é feita por expressões regulares. Se os layouts dos seus atestados variarem muito, ou se os campos não estiverem sempre no mesmo formato, a lógica de parsing pode precisar de ajustes ou de uma solução mais avançada de PLN (como um modelo de Reconhecimento de Entidades Nomeadas - NER).
    Otimização: Para uso em produção, você pode considerar otimizações adicionais como quantização do modelo ou uso de frameworks de inferência otimizados (ex: ONNX Runtime).
    Logging: O diretório logs montado no Docker pode ser usado com ferramentas como TensorBoard para visualizar o progresso do treinamento.

Se surgir qualquer dúvida ou problema durante a implantação ou uso, não hesite em perguntar!