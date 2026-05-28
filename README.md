# Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG (Retrieval-Augmented Generation) que ingere um PDF no PostgreSQL com pgVector e permite consultas via CLI, respondendo apenas com base no conteúdo do documento.

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- API Key da OpenAI e/ou Google Gemini

## Setup

1. Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

Edite o `.env` e preencha a API key do provedor desejado (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`).

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Ingestão do PDF

```bash
# Com OpenAI (padrão)
python src/ingest.py

# Com Gemini
python src/ingest.py --provider gemini
```

### 3. Chat interativo

```bash
# Com OpenAI (padrão)
python src/chat.py

# Com Gemini
python src/chat.py --provider gemini
```

## Exemplo de uso

```
Chat RAG - Digite 'sair' para encerrar

PERGUNTA: Qual o faturamento da empresa?
RESPOSTA: O faturamento foi de 10 milhões de reais.
---

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
---

PERGUNTA: sair
```

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py         # Ingestão do PDF no banco vetorial
│   ├── search.py         # Busca semântica + chamada à LLM
│   └── chat.py           # CLI interativo
├── document.pdf          # PDF para ingestão
└── README.md
```
