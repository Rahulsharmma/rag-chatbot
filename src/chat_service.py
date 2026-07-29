from langchain_groq import ChatGroq

from src.router import route_question
from src.rag_chain import get_rag_response


def get_general_response(question: str):

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )

    system_prompt = """
You are a helpful general-purpose AI assistant.

Answer the user's question clearly and accurately.

The question is not related to the company's
private document knowledge base.
"""

    response = llm.invoke(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                question,
            ),
        ]
    )

    return {
        "answer": response.content,
        "route": "general",
        "sources": [],
    }


def chat_service(question: str):

    route = route_question(
        question
    )

    print(
        f"Question route: {route}"
    )

    if route == "general":

        return get_general_response(
            question
        )

    elif route == "document":

        result = get_rag_response(
            question
        )

        result["route"] = "document"

        return result

    else:

        return {
            "answer": (
                "I was unable to determine "
                "how to answer your question."
            ),
            "route": "unknown",
            "sources": [],
        }