# Sistema de Extração de Dados de Atestados Médicos com TrOCR

Este projeto implementa uma solução para extração automática de campos específicos de atestados médicos em formato JPG utilizando o modelo **TrOCR** (Transformer-based Optical Character Recognition) da Microsoft.

A aplicação permite realizar o *fine-tuning* do modelo com uma base de dados personalizada e oferece uma interface gráfica intuitiva para upload de arquivos e controle do processo de treinamento e inferência. Todo o ambiente é empacotado em um container Docker para facilitar a implantação e garantir a reprodutibilidade.

---

## 📌 Visão Geral do Projeto

O core do projeto consiste em:

- **Fine-tuning do TrOCR**  
  Adaptação de um modelo TrOCR pré-treinado para o domínio específico de atestados médicos, permitindo um reconhecimento de texto otimizado.

- **Extração de Campos**  
  Após o OCR, uma etapa de pós-processamento com expressões regulares extrai campos-chave como:
  - Nome
  - Data
  - CID
  - Dias de afastamento
  - CRM do médico

- **Interface Gráfica (Streamlit)**  
  Aplicação web simples e funcional que facilita:
  - **Treinamento**: Upload de dados e configuração de hiperparâmetros.
  - **Inferência**: Upload de novos atestados para extração automática de texto e campos.

- **Containerização (Docker)**  
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
