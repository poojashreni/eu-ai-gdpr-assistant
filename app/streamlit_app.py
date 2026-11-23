"""Streamlit page for the EU AI Act & GDPR assistant. All logic lives in assistant.ui."""

import streamlit as st

from assistant import ui

st.set_page_config(page_title="EU AI Act & GDPR Assistant", page_icon="⚖️", layout="centered")


def _text_as_of() -> str:
    """Fetch once per session; retry on the next rerun while the API is unreachable."""
    if st.session_state.get("text_as_of", "unknown") == "unknown":
        st.session_state["text_as_of"] = ui.get_text_as_of()
    return st.session_state["text_as_of"]


def _pick_example(text: str) -> None:
    """Button callback: runs before the widgets are built, so it may set the input's value."""
    st.session_state["question"] = text
    st.session_state["pending"] = text


def _run(question: str) -> None:
    question = question.strip()
    if not question:
        st.session_state["result"] = ui.AskResult(error="Please type a question first.")
        return
    with st.spinner("Searching the legal texts..."):
        st.session_state["result"] = ui.ask(question)


st.title("EU AI Act & GDPR Assistant")
st.write(
    "Ask in German or English. Answers are grounded in the official texts "
    "and cite the exact provisions, with links to EUR-Lex."
)

st.text_input("Your question", key="question", max_chars=1000)
if st.button("Ask", type="primary"):
    _run(st.session_state["question"])

st.caption("Try an example:")
columns = st.columns(2)
for index, (language, text) in enumerate(ui.EXAMPLES):
    columns[index % 2].button(
        text,
        key=f"example_{index}",
        help=language,
        use_container_width=True,
        on_click=_pick_example,
        args=(text,),
    )

if "pending" in st.session_state:
    _run(st.session_state.pop("pending"))

result = st.session_state.get("result")
if result is not None:
    st.divider()
    if result.error:
        st.error(result.error)
    else:
        answer = result.answer
        if answer.refused:
            st.info(answer.answer)
        else:
            st.markdown(answer.answer)
            with st.expander("Sources", expanded=True):
                for line in ui.citation_links(answer):
                    st.markdown(f"- {line}")
        st.caption(ui.caption(answer))

st.divider()
st.caption(ui.footer(_text_as_of()))
