import streamlit as st

from dotenv import load_dotenv

from src.chat_service import chat_service


load_dotenv()


st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)


st.title(
    "🤖 AI Knowledge Assistant"
)

st.caption(
    "Ask general questions or questions "
    "about the company knowledge base."
)


with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )

    st.success(
        "Knowledge base loaded"
    )

    st.divider()

    st.subheader(
        "Routing"
    )

    st.write(
        """
        **General questions**

        → Groq LLM

        **Document questions**

        → ChromaDB + Groq
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    file_name = source.get(
                        "file_name",
                        "Unknown",
                    )

                    page = source.get(
                        "page"
                    )

                    if page is not None:

                        st.write(
                            f"📄 {file_name} "
                            f"— Page {page + 1}"
                        )

                    else:

                        st.write(
                            f"📄 {file_name}"
                        )


question = st.chat_input(
    "Ask a question..."
)


if question:

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            try:

                result = chat_service(
                    question
                )

                answer = result[
                    "answer"
                ]

                route = result[
                    "route"
                ]

                sources = result.get(
                    "sources",
                    [],
                )

                st.markdown(
                    answer
                )

                if route == "document":

                    st.caption(
                        "🔎 Answered using "
                        "the knowledge base"
                    )

                elif route == "general":

                    st.caption(
                        "🧠 Answered using "
                        "Groq general knowledge"
                    )

                if sources:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for source in sources:

                            file_name = source.get(
                                "file_name",
                                "Unknown",
                            )

                            page = source.get(
                                "page"
                            )

                            if page is not None:

                                st.write(
                                    f"📄 {file_name} "
                                    f"— Page "
                                    f"{page + 1}"
                                )

                            else:

                                st.write(
                                    f"📄 {file_name}"
                                )

            except Exception as e:

                answer = (
                    "Sorry, something went wrong."
                )

                route = "error"

                sources = []

                st.error(
                    str(e)
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "route": route,
            "sources": sources,
        }
    )