from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)
documents = [
    "virat kohli is a good cricketer",
    "MS Dhoni is a great wicket keeper",
    "Sachin Tendulkar is an overrated batsman",
    "Rohit Sharma is a nice captain",
    "Jasprit bumrah is a bowler"
]    
query = "tell me about Virat Kohli"
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)
scores = cosine_similarity([query_embedding],doc_embeddings)[0]
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is ",score)
