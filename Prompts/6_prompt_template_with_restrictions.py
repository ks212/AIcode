import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=hf_token,
    task="text-generation" 
)
st.title("Research Tool") 
paper_input = st.text_input("Enter your input")


template = PromptTemplate(
    input_variables=["paper_input"],
    template="""
Give a summary of {paper_input} in **JSON** format with keys:
- "brief"
- "key_points"
Limit each section to 20 words.
"""
)

prompt=template.invoke({
    'paper_input':paper_input,
    
})
model = ChatHuggingFace(llm=llm)
if st.button('Click'):
    response=model.invoke(prompt)
    st.write(response.content)