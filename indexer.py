"""Indexation des règles de la boutique dans ChromaDB
(à lancer une fois avant l'agent).

Exécution :
    uv run indexer.py
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

FICHIER_PDF = os.getenv(
    "FICHIER_PDF",
    "politique_boutique.pdf"
)

EMBEDDINGS = os.getenv(
    "EMBEDDINGS",
    "gemini-embedding-001"
)

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    "./chroma_db"
)

COLLECTION = os.getenv(
    "COLLECTION",
    "politique_boutique"
)


# =========================================================
# OUVERTURE DE LA BASE VECTORIELLE
# =========================================================

def ouvrir_base():
    """Ouvre la base vectorielle Chroma
    utilisée aussi par l'agent.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDINGS
    )

    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


# =========================================================
# INDEXATION
# =========================================================

def main():

    # =====================================================
    # 1. CHARGEMENT DU PDF
    # =====================================================

    loader = PyPDFLoader(
        FICHIER_PDF
    )

    docs = loader.load()

    print(
        "Pages chargées :",
        len(docs)
    )


    # =====================================================
    # 2. DECOUPAGE (CHUNKING)
    # =====================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        docs
    )

    print(
        "Chunks :",
        len(chunks)
    )


    # =====================================================
    # 3 + 4. EMBEDDINGS GEMINI + STOCKAGE CHROMADB
    # =====================================================

    db = ouvrir_base()

    # Vide la collection pour éviter les doublons
    db.reset_collection()

    db.add_documents(
        chunks
    )

    print(
        f"Index ChromaDB créé : "
        f"{len(chunks)} chunks dans "
        f"'{COLLECTION}' ({CHROMA_DIR})"
    )


    # =====================================================
    # 5. TEST RAPIDE DU RETRIEVER
    # =====================================================

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3
        }
    )

    question = (
        "Puis-je retourner un produit après 20 jours ?"
    )

    print(
        f"\nTest : {question}"
    )

    for doc in retriever.invoke(
        question
    ):

        print(
            "   •",
            doc.page_content
            .replace("\n", " ")[:120],
            "..."
        )


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":
    main()