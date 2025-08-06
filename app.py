import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Set Gemini API Key from secrets or environment
def get_api_key():
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY", None)

api_key = get_api_key()

if not api_key:
    st.error("Google Gemini API key not found. Please add it to Streamlit secrets or set as an environment variable.")
    st.stop()

os.environ["GEMINI_API_KEY"] = api_key

st.set_page_config(
    page_title="Gemini PDF Q&A Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Gemini PDF Question Answering Chatbot")

# Sidebar for instructions and API key display
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1. Upload a PDF file.
        2. The app will process and index the document.
        3. Ask questions related to your PDF.
        4. View answers and source excerpts.
        """
    )
    st.markdown("---")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file:
    if 'processed_docs' not in st.session_state:
        with st.spinner("Loading and processing document..."):
            with open("uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("uploaded.pdf")
            documents = loader.load()
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
            vectorstore = Chroma.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key),
                retriever=retriever,
                return_source_documents=True,
            )
            st.session_state.processed_docs = {
                "qa_chain": qa_chain,
                "docs": documents
            }
        st.success("Document processed successfully!")
    
    st.subheader("Ask a question about the PDF")
    query = st.text_input("Enter your question:", key="user_query")
    ask_button = st.button("Get Answer")

    if ask_button:
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                response = st.session_state.processed_docs["qa_chain"]({"query": query})
            st.markdown("### Answer")
            st.write(response["result"])

            with st.expander("Show source document excerpts"):
                for i, doc in enumerate(response["source_documents"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.write(doc.page_content)
else:
    st.info("📥 Please upload a PDF file to begin.")
