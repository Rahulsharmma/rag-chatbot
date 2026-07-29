from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import get_vector_store


def get_rag_response(question: str):

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        },
    )

    retrieved_docs = retriever.invoke(
        question
    )

    if not retrieved_docs:

        return {
            "answer": (
                "I could not find relevant information "
                "in the knowledge base."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful company knowledge assistant.

Answer the user's question using ONLY the
provided context.

If the answer cannot be found in the context,
say:

"I could not find this information in the
provided documents."

Do not make up information.

Context:
----------------
{context}
----------------
""",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = []

    for document in retrieved_docs:

        source = document.metadata.get(
            "source",
            "Unknown",
        )

        page = document.metadata.get(
            "page",
            None,
        )

        file_name = document.metadata.get(
            "file_name",
            source,
        )

        source_info = {
            "file_name": file_name,
            "source": source,
            "page": page,
        }

        if source_info not in sources:

            sources.append(
                source_info
            )

    return {
        "answer": response.content,
        "sources": sources,
    }