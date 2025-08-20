from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # You can change this to any supported HF model
)

documents = [
    "virat kohli is a good cricketer",
    "MS Dhoni is a great wicket keeper",
    "Sachin Tendulkar is an overrated batsman",
    "Rohit Sharma is a nice captain",
    "Jasprit bumrah is a bowler"
]    
query = "tell me about Virat Kohli"

# Get embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Convert to numpy arrays
doc_embeddings = np.array(doc_embeddings)
query_embedding = np.array(query_embedding).reshape(1, -1)

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)

# Output similarities
print("Similarity scores (higher = more similar):")
for i, score in enumerate(similarities[0]):
    print(f"{score:.4f} → {documents[i]}")
