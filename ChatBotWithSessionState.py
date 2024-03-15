import streamlit as st
import os
from constants import openai_key

from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = openai_key
llm = ChatOpenAI(temperature=0.7,model="gpt-3.5-turbo")

#in start of session just set the system message
if 'flowmessage' not in st.session_state: 
    st.session_state['flowmessage'] = [
        SystemMessage(content="YOU are an Ai assistent")
    ]

def bot_response(question):

    st.session_state['flowmessage'].append(HumanMessage(content=question))
    response = llm.invoke(st.session_state["flowmessage"])      #to get response pass the whole session["flowmessage"]

    st.session_state['flowmessage'].append(AIMessage(content=response.content))    #passing the response from the ai back to the seassion using the AIMessage
    
    return response.content

st.set_page_config(
    page_title="Convosational Chatbot"
)

st.header("Lets have a conv.")

input = st.text_input("Query",key=input)
submit = st.button("Generate")

if submit:
    st.subheader("Response")
    response = bot_response(input)
    st.write(response)


