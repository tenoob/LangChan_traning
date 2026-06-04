import requests
import streamlit as st

def openai_response(input_text):
    response = requests.post("http://localhost:8005/gptHelp/invoke",
                             json={
                                 'input':{'topic':input_text}
                             })
    return response.json()['output']['content']


def gemma_response(input_text):
    response = requests.post("http://localhost:8005/funFact/invoke",
                             json={'input':{'topic':input_text}}
                             )
    return response.json()['output']


st.title("Api deployment demo")
input_text1 = st.text_input("ask about something")
input_text2 = st.text_input("fun fact about")

if input_text1:
    st.write(openai_response(input_text1))

if input_text2:
    st.write(gemma_response(input_text2))