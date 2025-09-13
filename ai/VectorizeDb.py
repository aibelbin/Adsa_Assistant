from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

file = "ai/data.csv"

loader = CSVLoader(file_path=file, encoding="utf-8")
data = loader.load()

persistent_dir = "./chromaFor_db"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = Chroma.from_documents(
    documents=data,
    collection_name = "Asthra10.0",
    embedding = embeddings,
    persist_directory = persistent_dir,
)
print('Created')


