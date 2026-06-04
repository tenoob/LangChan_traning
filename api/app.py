from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

import uvicorn
import os


load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

app = FastAPI(
    title = "Langchain_server1",
    version="0.1",
    description="Api server for multiple LLMs"
)


gpt = ChatOpenAI()
llama = OllamaLLM(model="gemma3")

promt1 = ChatPromptTemplate.from_template("give me information about {topic} in 50 words")
promp2 = ChatPromptTemplate.from_template("tell me a fun fact about {topic}")

add_routes(
    app,
    promt1|gpt,
    path="/gptHelp"
)

add_routes(
    app,
    promp2|llama,
    path="/funFact"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8005)
    #access the routes using /{path}/playground 