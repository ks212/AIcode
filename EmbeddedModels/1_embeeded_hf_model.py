from langchain_huggingface import HuggingFaceEmbeddings

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = "delhi is the capital of india"
vector = embedding.embed_query(text)

print(str(vector))