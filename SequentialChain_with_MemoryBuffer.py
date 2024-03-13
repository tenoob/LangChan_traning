import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

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

example_prompt_3 = PromptTemplate(
    input_variables=['dob'],
    template="5 major event happened around {dob} in india"
)

#memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

#llms
llm = ChatOpenAI(temperature=0.7)
chain_1 = LLMChain(
    llm=llm, 
    prompt=example_prompt_1 , 
    verbose=True,
    output_key='person',
    memory=person_memory)

chain_2 = LLMChain(
    llm=llm, 
    prompt=example_prompt_2 , 
    verbose=True,
    output_key='dob',
    memory=dob_memory)

chain_3 = LLMChain(
    llm=llm, 
    prompt=example_prompt_3 , 
    verbose=True,
    output_key='description',
    memory=descr_memory)


main_chain = SequentialChain(
    chains=[chain_1,chain_2,chain_3],
    verbose=True,
    input_variables=['name'],
    output_variables=['person','dob','description']
    )

if input_text:
    st.write(main_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('DOB'): 
        st.info(dob_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)
