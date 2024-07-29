from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

api_key="96a616c0-f376-4b04-9fb3-f5ab738f25e4"

index_name="hybrid-search-langchain-pinecone"
## initialize the Pinecone client
pc=Pinecone(api_key=api_key)

#create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index=pc.Index(index_name)

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


bm25_encoder=BM25Encoder().default()

sentences=[
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

]

## tfidf values on these sentence
bm25_encoder.fit(sentences)

## store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)

retriever.add_texts(
    [
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

]
)

retriever.invoke("What city did i visit first")


