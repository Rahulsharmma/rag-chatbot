from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


def load_documents(data_dir: str):
    """
    Load all supported documents from a directory.

    Supported:
        .pdf
        .txt
        .docx

    Returns:
        List[Document]
    """

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {data_dir}"
        )

    if not data_path.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {data_dir}"
        )

    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }

    all_documents = []

    for file_path in sorted(data_path.rglob("*")):

        # Ignore directories
        if not file_path.is_file():
            continue

        # Ignore hidden files
        if file_path.name.startswith("."):
            continue

        extension = file_path.suffix.lower()

        # Ignore unsupported file types
        if extension not in loader_map:
            print(
                f"Skipping unsupported file: {file_path.name}"
            )
            continue

        print(f"Loading: {file_path.name}")

        try:
            loader_class = loader_map[extension]

            loader = loader_class(str(file_path))

            documents = loader.load()

            # Add additional metadata
            for document in documents:
                document.metadata["file_name"] = file_path.name
                document.metadata["file_type"] = extension

            all_documents.extend(documents)

            print(
                f"  Loaded {len(documents)} document object(s)"
            )

        except Exception as e:
            print(
                f"  Failed to load {file_path.name}: {e}"
            )

    print(
        f"\nTotal loaded document objects: "
        f"{len(all_documents)}"
    )

    return all_documents