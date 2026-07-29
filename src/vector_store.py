from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "company_knowledge_base"


def get_embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def create_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    print(
        f"Created {len(chunks)} chunks "
        f"from {len(documents)} document objects."
    )

    return chunks


def create_vector_store(chunks):

    embeddings = get_embedding_model()

    Path(CHROMA_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(
        f"Vector store created at: {CHROMA_DIR}"
    )

    return vector_store


def get_vector_store():

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store