import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
st.title("#Demo 1: LangChan with OpenAi api")
input_text = st.text_input("text to search using api")

llm = ChatOpenAI(
    temperature=0.8)

if input_text:
    #generic search on the given input text
    st.write(llm.invoke(input_text))

