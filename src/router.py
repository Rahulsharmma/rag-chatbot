from typing import Literal

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


class RouteDecision(BaseModel):

    route: Literal[
        "general",
        "document"
    ] = Field(
        description=(
            "Whether the question should be "
            "answered using general knowledge "
            "or company documents."
        )
    )


def route_question(question: str) -> str:

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    structured_llm = llm.with_structured_output(
        RouteDecision
    )

    system_prompt = """
You are a routing classifier.

Decide whether the user's question should be
answered using general LLM knowledge or the
company document knowledge base.

Choose "document" when the question asks about:

- company policies
- company rules
- employee benefits
- leave policy
- termination
- notice period
- employee handbook
- information from uploaded documents

Choose "general" when the question is unrelated
to the company documents.

Examples:

Question:
What is the capital of India?

Route:
general

Question:
What is Python?

Route:
general

Question:
What is the termination policy?

Route:
document

Question:
How many days of leave are allowed?

Route:
document
"""

    decision = structured_llm.invoke(
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

    return decision.route