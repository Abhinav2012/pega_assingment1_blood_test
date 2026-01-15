from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import os

token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=token,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

query = "what happens if WBC count is low in blood test"
results = vectorstore.similarity_search(query, k=5)

for i, doc in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)