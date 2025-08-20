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
style_input = st.selectbox("select Explanation style",["Beginner-Friendly", "Technical", "Code-Oriented","MAthematical"])
length_input = st.selectbox("Select Explanation Length",["Short ( 1-2 Paragraph)", "Medium (3.5 paragraphs)","Long (detailed Explanation)"])
role=st.selectbox("select Your Role",["CEO","Doctor","Cricketer"])

template=PromptTemplate(template="""
 please summarize Details about titled "{paper_input}" with the following specification:
explanation style: {style_input}
explanation length: {length_input}
role based input:{role}
1.Provide details according to role if 
{role} is Doctor and {paper_input} is related to Medical then share only medical details.
{role} is CEO and {paper_input} is related to Industry then only share the details. 
{role} is Cricketer and {paper_input} is Cricket  then share only Cricket details.                                                
otherwise say i don't know.
2. Mathematical Details:
-include relevant mathematical equations if present in the paper.
-Explain the mathematical concepts using simple, intuitive code snippets where applicable.
3. Analogies:
-use relatable analogies to simplify complex ideas.
if certain information is not available in the paper, respond with:"insufficient information available" instead of guessing.
Ensure the summary is clear, accurate, and aligned with the provided style and length.

 """,
 input_variables=['paper_input','style_input','length_input','role'])

prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input,
    'role':role
})
model = ChatHuggingFace(llm=llm)
if st.button('Click'):
    response=model.invoke(prompt)
    st.write(response.content)