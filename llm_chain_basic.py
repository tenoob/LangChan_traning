import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
st.title("#Demo 1: LangChan with OpenAi api")
input_text = st.text_input("text to search using api")

# templates
example_prompt = PromptTemplate(
    input_variables=['name'],
    template="who is this person: {name}"
)

#llms
llm = ChatOpenAI(temperature=0.7)
chain = LLMChain(
    llm=llm, 
    prompt=example_prompt , 
    verbose=True)

if input_text:
    st.write(chain.run(input_text))
