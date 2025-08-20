from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
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
chat_history = [
    SystemMessage(content='you are a helpful assistant')
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)