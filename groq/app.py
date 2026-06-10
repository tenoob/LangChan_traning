import streamlit as st
import os
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

#load groq key 
groq_api_key = os.environ['GROQ_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model ="embeddinggemma:300m")
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/oss/python/integrations/embeddings/ollama")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_spliiter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=1000)
    st.session_state.documents = st.session_state.text_spliiter.split_documents(
        st.session_state.docs
    )

    st.session_state.vectors = FAISS.from_documents(
        st.session_state.documents,st.session_state.embeddings
    )

st.title("Groq Demo")
llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question based on the context provided.
The respone should be accurate based on the context.
<context>
{context}
</context>
Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever,document_chain)

prompt = st.text_input("Give the question")

if prompt:
    start = time.process_time()
    response = retriever_chain.invoke({"input":prompt})
    st.write(f"Execution Time: {time.process_time() - start}")
    st.write(response['answer'])

    with st.expander("Document search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("******************************")

