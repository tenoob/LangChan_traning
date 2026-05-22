from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"


#prompt templete
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an assistent. Help with the questions"),
        ("user,Question:{question}")
    ]
)

#streamlit framework
st.title("Test 1")
input_text = st.text_input("Give input")

#openai
llm = Ollama(model="gemma3")
output_parser = StrOutputParser()
chain = prompt | llm |output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))



