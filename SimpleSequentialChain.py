import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
st.title("#Demo 1: LangChan with OpenAi api")
input_text = st.text_input("text to search using api")

# templates
example_prompt_1 = PromptTemplate(
    input_variables=['name'],
    template="who is this person: {name}"
)

example_prompt_2 = PromptTemplate(
    input_variables=['person'],
    template="where was {person} born"
)

#llms
llm = ChatOpenAI(temperature=0.7)
chain_1 = LLMChain(
    llm=llm, 
    prompt=example_prompt_1 , 
    verbose=True,
    output_key='person')

chain_2 = LLMChain(
    llm=llm, 
    prompt=example_prompt_2 , 
    verbose=True,
    output_key='dob')

main_chain = SimpleSequentialChain(chains=[chain_1,chain_2],verbose=True)

if input_text:
    st.write(main_chain.run(input_text))
