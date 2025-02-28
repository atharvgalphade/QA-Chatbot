from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
import streamlit as st 
import os
from dotenv import load_dotenv

load_dotenv() #this will load the virtual env and libraries install in that env

#langsmith tracking
os.environ['LANGCHAIN_API_KEY']='Langchain_api_key'
os.environ['LANGCHAIN_TRACING_V2']='True'
os.environ['LANGCHAIN_PROJECT']="Q&A Chatbot with Ollama"
# Loads API keys and configurations for LangSmith tracking (used for debugging and monitoring LangChain applications).
# Sets LANGCHAIN_TRACING_V2 = "true" to enable logging and debugging of responses.

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please give response to the user queries"),
        ("user","Question:{question}")
    ]
)
# "system" message sets the AI's behavior.
# "user" message contains the actual user input, formatted as Question:{question} (a placeholder).

def generate_response(question,engine,temperature,max_tokens):
    llm=Ollama(model=engine)
    output_parser=StrOutputParser()
    pipeline=prompt | llm | output_parser
    answer=pipeline.invoke({'question':question})
    return answer

# This function takes the user's question and parameters (engine, temperature, max_tokens) as input.
# Ollama(model=engine) loads the chosen open-source model (e.g., "gemma2" or "llama3.2").
# Pipeline:
# prompt | llm | output_parser
# It first applies the prompt template, then passes it to the Ollama model, and finally ensures output is in a string format.
# chain.invoke({'question': question}) executes the pipeline and generates a response.

# title of the app
st.title("Q&A chatbot with selection of open source model")

engine=st.sidebar.selectbox("Select and Ollama model",["gemma2","llama3"])

#adjust response parameter
temperature=st.sidebar.slider('Temperature',min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider('Max Tokens',min_value=50,max_value=300,value=100)

#main interface for chat
st.write("Hey User! Please ask your question")
question=st.text_input("You: ")

if question:
    response=generate_response(question,engine,temperature, max_tokens)
    st.write(response)
    
else:
    st.write("Please provide the user input")
    
