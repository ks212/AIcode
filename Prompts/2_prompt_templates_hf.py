import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate


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

template = PromptTemplate(
    template="""
please summarize the research paper titled "{paper_input}" with the following specification:
explanation style: {style_input}
explanation length: {length_input}
1. Mathematical Details:
-include relevant mathematical equations if present in the paper.
-Explain the mathematical concepts using simple, intuitive code snippets where applicable.
2. Analogies:
-use relatable analogies to simplify complex ideas.
if certain information is not available in the paper, respond with:"insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.

""",
input_variables=['paper_input','style_input','length_input'],
validate_template=True
)

prompt = template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('click'):
    response = model.invoke(prompt)
    st.write(response.content)
