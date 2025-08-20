from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=hf_token,
    task="text-generation"
)
model = ChatHuggingFace(llm=llm1)
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    result = model.invoke(user_input)
    print("AI:", result.content)