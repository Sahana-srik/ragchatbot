import streamlit as st
from app import ask_question

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🤖 RAG Chatbot")

# --------------------------------
# Initialize chat memory
# --------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------
# Display chat history
# --------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------
# Chat input (like ChatGPT)
# --------------------------------
prompt = st.chat_input("Ask something from the PDFs...")

if prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation context
    history_text = ""
    for m in st.session_state.messages:
        if m["role"] == "user":
            history_text += f"User: {m['content']}\n"
        else:
            history_text += f"Assistant: {m['content']}\n"

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(prompt, history_text)
            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
