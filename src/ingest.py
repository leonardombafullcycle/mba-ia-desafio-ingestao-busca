import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")


def get_embeddings(provider):
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-2")
        return GoogleGenerativeAIEmbeddings(model=model)
    else:
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)


def ingest_pdf(provider="openai"):
    docs = PyPDFLoader(PDF_PATH).load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)

    embeddings = get_embeddings(provider)
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    ids = [f"doc-{i}" for i in range(len(splits))]
    store.add_documents(documents=splits, ids=ids)
    print(f"Ingestão concluída: {len(splits)} chunks armazenados com provider '{provider}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    args = parser.parse_args()
    ingest_pdf(args.provider)
