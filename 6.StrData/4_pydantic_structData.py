from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import json
#from pydantic import BaseModel,Field
import os

load_dotenv()
hf_token = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    huggingfacehub_api_token=hf_token,
    task="text-generation"
)
model = ChatHuggingFace(llm=llm1)

#class Review(BaseModel):
   # key_themes : list[str] = Field(description='write down all the key themes discussed in the review')
   # summary: str = Field(description="A brief summary of review")

    #key_themes: Annotated[list[str],'write down all the key themes discussed in the review']
    #summary: Annotated[str,"A brief summary of review"]
    #sentiment: Annotated[Literal["pos","neg"],"return sentiment of the review either positive, negitive"]
   # pros: Annotated[Optional[list[str]],'write down all the pros inside the list']

#structured_model = model.with_structured_output(Review)
#result = structured_model.invoke("""
#the hardware is great but software feels bloated. there are too many pre-installed apps that i cant remove. also UI looks outdated""")
prompt="""
the hardware is great but software feels bloated. there are too many pre-installed apps that i cant remove. also UI looks outdated"""
result = model.invoke(prompt)
print(result)

# Optional: Try parsing the output as JSON (if it's well-formed)
try:
    json_output = json.loads(result.content)
    print("Parsed Output:\n", json_output)
except json.JSONDecodeError:
    print("Could not parse response as JSON. Here's the raw output again:\n")
    print(result.content)