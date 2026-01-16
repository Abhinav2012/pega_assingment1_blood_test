import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader

load_dotenv(override=True)


docs_path="docs"

loader = DirectoryLoader(
path=docs_path,
glob="*.txt",
loader_cls=TextLoader,
loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

if len(documents) == 0:
    raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")

for i, doc in enumerate(documents[:2]):  # Show first 2 docunts
    print(f"\nDocument {i+1}:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content length: {len(doc.page_content)} characters")
    print(f"  Content preview: {doc.page_content[:100]}...")
    print(f"  metadata: {doc.metadata}")

chunk_size = 1000
chunk_overlap = 200

text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap
)

chunks = text_splitter.split_documents(documents)

if chunks:

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:")
        print(chunk.page_content)
        print("-" * 50)
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")

print("Trying to make embeddings")

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=token,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
test_vector =  embeddings.embed_query("Is this working?")
print(f"Vector Length: {len(test_vector)}")

# Create vector store - this embeds all chunks automatically
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted.")

