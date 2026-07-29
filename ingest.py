from dotenv import load_dotenv

from src.document_loader import load_documents
from src.vector_store import (
    create_chunks,
    create_vector_store,
)


load_dotenv()


DATA_DIR = r"C:\Users\DELL\OneDrive\Desktop\rag_data"


def main():

    print("=" * 60)
    print("STARTING DOCUMENT INGESTION")
    print("=" * 60)

    print("\n[1/3] Loading documents...")

    documents = load_documents(
        DATA_DIR
    )

    if not documents:

        print(
            "No documents found."
        )

        return

    print("\n[2/3] Creating chunks...")

    chunks = create_chunks(
        documents
    )

    print(
        "\n[3/3] Creating vector database..."
    )

    create_vector_store(
        chunks
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "INGESTION COMPLETED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()