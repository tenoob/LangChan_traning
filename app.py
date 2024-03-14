from langchain_openai import OpenAI
from constants import openai_key

import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = openai_key


def openai_response(question):
    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.7)
    response = llm.invoke(question)
    return response

st.set_page_config(page_title="QA")
st.header("Davinci QA app")

input = st.text_input("Query",key=input)
submit = st.button("Generate")

if submit:
    st.subheader("Response")
    response = openai_response(input)
    st.write(response)