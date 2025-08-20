import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=hf_token,
     task="text-generation" 
)
st.header("Research Tool")

#query = st.text_input("Ask a question:")
model = ChatHuggingFace(llm=llm)

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "Bert:Pre-training of Deep Bidirectional Transformers","GPT-3:Language models are Few-Shot Learners"])
style_input = st.selectbox("select Explanation style",["Beginner-Friendly", "Technical", "Code-Oriented","MAthematical"])
length_input = st.selectbox("Select Explanation Length",["Short ( 1-2 Paragraph)", "Medium (3.5 paragraphs)","Long (detailed Explanation)"])

template = load_prompt('template.json')


if st.button('click'):
    chain = template | model

    response = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})
    st.write(response.content)
