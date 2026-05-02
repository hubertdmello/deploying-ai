import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from __init__ import initialize_openai_client, API_GATEWAY_URL
from prompts import DRUCKER_SYSTEM_PROMPT


_PDF_PATH = Path(__file__).parent / "../../02_activities/documents/managing_oneself.pdf"
_CHROMA_PATH = Path(__file__).parent / "chroma_db"
_COLLECTION_NAME = "managing_oneself"
_drucker_collection = None


def _get_drucker_collection():
    global _drucker_collection
    if _drucker_collection is not None:
        return _drucker_collection

    ef = OpenAIEmbeddingFunction(
        api_key='any value',
        model_name="text-embedding-3-small",
        api_base=API_GATEWAY_URL,
        default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')}
    )

    chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef
    )

    if collection.count() == 0:
        loader = PyPDFLoader(str(_PDF_PATH))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        collection.add(
            documents=[c.page_content for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"page": c.metadata.get("page", 0)} for c in chunks]
        )

    _drucker_collection = collection
    return _drucker_collection


def ask_drucker(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a question about Peter Drucker's 'Managing Oneself'."

    try:
        collection = _get_drucker_collection()
        client = initialize_openai_client()

        results = collection.query(query_texts=[message], n_results=3)
        retrieved_chunks = results["documents"][0]
        context = "\n\n---\n\n".join(retrieved_chunks)

        messages = [{"role": "system", "content": DRUCKER_SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": f"Context from 'Managing Oneself':\n\n{context}\n\nQuestion: {message}"
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=messages
        )

        answer = response.choices[0].message.content
        return f"I retrieved this from the Drucker knowledge base using my RAG pipeline. Here's what I found:\n\n{answer}"

    except ValueError as e:
        return f"Configuration error: {str(e)}\n\nPlease ensure your OPENAI_API_KEY environment variable is set."
    except Exception as e:
        return f"Error querying the knowledge base: {str(e)}"
