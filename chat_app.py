# chat_app.py
import os
import base64
import streamlit as st
from strategy_rag import qa_chain

# --------------------------------------------------
# ✅ Streamlit Page Config (must be first)
# --------------------------------------------------
st.set_page_config(page_title="IIMA Strategy SMBL07 Assistant By Vivek", page_icon="📘", layout="wide")

# --------------------------------------------------
# ✅ Load Logo from data/
# --------------------------------------------------
logo_path = os.path.join("data", "Indian Institute of Management logo.jpeg")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/jpeg;base64,{logo_base64}" width="100">
            <h2 style="margin-left: 20px;">IIMA Strategy SMBL07 Assistant</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("⚠️ Logo not found. Please add it to `data/Indian Institute of Management logo.jpeg`")

st.markdown("---")

# --------------------------------------------------
# ✅ Chat History
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# ✅ Chat Input
# --------------------------------------------------
if prompt := st.chat_input("Ask me anything about Strategy..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            response = result.get("result", "⚠️ Sorry, I couldn’t generate a response.")
            st.markdown(response)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})
