import streamlit as st
import tempfile
import os
import shutil

from src.helper import load_pdf, splitting, embeds
from src.pipeline import retrieval

st.set_page_config(page_title="PDF Chat MVP", layout="centered")
st.title("HarisGPT, Let's Chat with PDF!!!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.sidebar.header("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            try:
                # 1. Save Temp File
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 2. Load
                documents = load_pdf(tmp_path)
                
                # 3. Split
                chunks = splitting(documents)

                if os.path.exists("./chroma_db"):
                    try:
                        shutil.rmtree("./chroma_db")
                    except Exception:
                        pass 
                st.session_state.vectorstore = embeds(chunks)
                
                os.remove(tmp_path)
                
                st.sidebar.success("PDF Indexed! Chat now.")
                
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    if st.session_state.vectorstore:
        with st.spinner("Thinking..."):
            answer = retrieval(st.session_state.vectorstore, prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    else:
        st.warning("Please upload and process a PDF first.")