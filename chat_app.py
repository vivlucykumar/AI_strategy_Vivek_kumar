# # chat_app.py
# import streamlit as st
# from strategy_rag import qa_chain  # Import your working RAG pipeline
# import streamlit as st
# import base64
# import os

# logo_path = os.path.join("data", "Indian Institute of Management logo.jpeg")

# with open(logo_path, "rb") as img_file:
#     logo_base64 = base64.b64encode(img_file.read()).decode()
# # -------------------------
# # Streamlit Page Config
# # -------------------------
# st.set_page_config(page_title="IIMA Strategy SMBL07 Assistant By Vivek Kumar", page_icon="📊", layout="centered")


# ALLOWED_EMAILS = {
#     "viv1989kumar@gmail.com",
#     "admin@gmail.com",
#     "user@gmail.com",
#     "chitra@gmail.com",  # Add all permitted emails, case-insensitive
# }

# def login():
#     st.markdown("## Login")
#     email = st.text_input("Email", key="login_email")
#     password = st.text_input("Password", type="password", key="login_password")
#     login_btn = st.button("Login")
#     login_status = False
#     if login_btn:
#         if email.strip().lower() in [e.lower() for e in ALLOWED_EMAILS] and password:  # Basic password presence
#             st.session_state.logged_in = True
#             st.success("Login successful! Click Login again to Enter....")
#             login_status = True
#         else:
#             st.error("Access denied: Email not authorized.")
#     return login_status

# # Ensure session state
# if "logged_in" not in st.session_state or not st.session_state.logged_in:
#     logged_in = login()
#     st.stop()  # This prevents running the rest of the app unless logged in

# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: none;
#     }}
#     .stApp::before {{
#         content: "";
#         position: fixed;
#         top: 0; left: 0; width: 100vw; height: 100vh;
#         background: url("data:image/jpeg;base64,{logo_base64}") center center no-repeat;
#         background-size: 60vw 60vw;
#         opacity: 0.07;
#         z-index: -1;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# st.title("📊 IIMA Strategy SMBL07 Assistant By Vivek Kumar")
# st.write("Ask me any strategy-related question regarding out topics (Porter, SWOT, Corporate Strategy, etc.).")

# # -------------------------
# # Chat History
# # -------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# # -------------------------
# # User Input
# # -------------------------
# if prompt := st.chat_input("Type your question here..."):
#     # Store user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate assistant reply
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             result = qa_chain.invoke({"input": prompt})
#             response = result.get("answer") or result.get("output_text", "⚠️ No response generated.")
#             st.markdown(response)

#     # Store assistant message
#     st.session_state.messages.append({"role": "assistant", "content": response})

# # ############################################################
# chat_app.py
# chat_app.py

import streamlit as st
import base64
import os
import sys

# Patch sqlite3 for Streamlit Cloud compatibility
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from strategy_rag import qa_chain # Import your working RAG pipeline

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="IIMA Strategy SMBL07 Assistant By Vivek Kumar", page_icon="📊", layout="centered")

# Basic login function
ALLOWED_EMAILS = {
    "viv1989kumar@gmail.com",
    "admin@gmail.com",
    "user@gmail.com",
    "chitra@gmail.com",
}

def show_login_page():
    st.markdown("## Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    login_btn = st.button("Login")
    if login_btn:
        if email.strip().lower() in [e.lower() for e in ALLOWED_EMAILS] and password:
            st.session_state.logged_in = True
            st.success("Login successful! Please click 'Login' again.")
            st.experimental_rerun()
        else:
            st.error("Access denied: Email not authorized.")

def show_main_app():
    # Load and display logo in the background
    logo_path = os.path.join("data", "Indian Institute of Management logo.jpeg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()

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
    else:
        st.warning(f"⚠️ Logo file not found at {logo_path}. Skipping background image.")

    st.title("📊 IIMA Strategy SMBL07 Assistant By Vivek Kumar")
    st.write("Ask me any strategy-related question regarding our topics (Porter, SWOT, Corporate Strategy, etc.).")

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
                response = result.get("answer", "⚠️ No response generated.")
                st.markdown(response)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App Logic ---
# This new logic correctly handles the login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    show_main_app()
else:
    show_login_page()
