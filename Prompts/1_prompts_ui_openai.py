import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=hf_token,
     task="text-generation" 
)
st.title("LLM Chatbot")

query = st.text_input("Ask a question:")
model = ChatHuggingFace(llm=llm)

if st.button('click'):
    response = model.invoke(query)
    st.write(response.content)
