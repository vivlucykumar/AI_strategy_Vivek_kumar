# chat_app.py
import streamlit as st
from strategy_rag import qa_chain  # Import your working RAG pipeline
import streamlit as st
import base64

ALLOWED_EMAILS = {
    "viv1989kumar@gmail.com",
    "admin@gmail.com",
    "user@gmail.com",
    "chitra@gmail.com",  # Add all permitted emails, case-insensitive
}

def login():
    st.markdown("## Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    login_btn = st.button("Login")
    login_status = False
    if login_btn:
        if email.strip().lower() in [e.lower() for e in ALLOWED_EMAILS] and password:  # Basic password presence
            st.session_state.logged_in = True
            st.success("Login successful!")
            login_status = True
        else:
            st.error("Access denied: Email not authorized.")
    return login_status

# Ensure session state
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    logged_in = login()
    st.stop()  # This prevents running the rest of the app unless logged in


logo_path = r"C:\Vivek\Personal\Documents\IIMA Strategic mgt course\16_Business Ideas\AI_strategy\data\Indian Institute of Management logo.jpeg"

with open(logo_path, "rb") as img_file:
    logo_base64 = base64.b64encode(img_file.read()).decode()

st.set_page_config(page_title="IIMA Strategy SMBL07 Assistant By Vivek Kumar", page_icon="📊", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{
        background: none;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: url("data:image/jpeg;base64,{logo_base64}") center center no-repeat;
        background-size: 60vw 60vw;
        opacity: 0.07;
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="IIMA Strategy SMBL07 Assistant By Vivek Kumar", page_icon="📊", layout="centered")

st.title("📊 IIMA Strategy SMBL07 Assistant By Vivek Kumar")
st.write("Ask me any strategy-related question regarding out topics (Porter, SWOT, Corporate Strategy, etc.).")

# -------------------------
# Chat History
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User Input
# -------------------------
if prompt := st.chat_input("Type your question here..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"input": prompt})
            response = result.get("answer") or result.get("output_text", "⚠️ No response generated.")
            st.markdown(response)

    # Store assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
