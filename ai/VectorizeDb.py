from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os


file = "ai/data.csv"
persistent_dir = "./chromaFor_db"
collection_name = "Asthra10.0"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

loader = CSVLoader(file_path=file, encoding="utf-8")
data = loader.load()


if not os.path.exists(os.path.join(persistent_dir, "index")):
    loader = CSVLoader(file_path=file, encoding="utf-8")
    data = loader.load()
    vector_store = Chroma.from_documents(
        documents=data,
        collection_name=collection_name,
        embedding=embeddings,
        persist_directory=persistent_dir
    )  
    print("Created DB")
else:
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persistent_dir
    )
    print("Loaded existing DB")

