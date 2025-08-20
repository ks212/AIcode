from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm1= HuggingFacePipeline.from_model_id(
    model_id='tiiuae/falcon-7b-instruct',
    task='text-generation',
    pipeline_kwargs=dict(
        temprature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm1)
result = model.invoke("what is the capital of india")

print(result.content)
