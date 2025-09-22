import os
import asyncio
import re  
import sounddevice as sd
from dotenv import load_dotenv
from kokoro import KPipeline
from RealtimeSTT import AudioToTextRecorder

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # added



# sd.default.device = "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"  #test
# this is just a test code to see wher the iported modules are going :) 


file = "ai/data.csv"
persistent_dir = "./chromaFor_db"
collection_name = "Asthra10.0"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

loader = CSVLoader(file_path=file, encoding="utf-8")
data = loader.load()

if not os.path.exists(os.path.join(persistent_dir, "index")):
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


retriever = vector_store.as_retriever()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    streaming=False,
    api_key=api_key
)

template = """You are a helpful and professional AI receptionist for the Department of Artificial Intelligence at St. Joseph's College of Engineering and Technology, Palai. Your name is 'Astra'.
Your purpose is to assist students, faculty, and visitors by answering their questions accurately based on the official information provided to you.

## Instructions:
1.  **Strictly Grounded:** Base your entire answer ONLY on the information found in the provided CONTEXT. Do not use any outside knowledge.
2.  **Synthesize, Don't Just Repeat:** Combine information from the context to form a coherent, easy-to-understand answer. Don't just copy-paste sentences.
3.  **Be Conversational:** Your tone should be friendly and professional. Keep your answers concise and clear, as they will be spoken aloud.
4.  **Handle Missing Information:** If the CONTEXT does not contain the answer to the question, politely say so. You can say something like, "I'm sorry, I don't have that specific information right now," or "I can't seem to find details on that."
5.  **Handle Greetings & Small Talk:** If the user says "hello" or "thank you," respond with a brief, polite greeting or acknowledgment.

CONTEXT:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs:
        return ""
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[Doc {i}] {d.page_content.strip()}")
    return "\n\n".join(parts)

rag_chain = (
    {
        "context": retriever ,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    recorder = AudioToTextRecorder()
    pipeline = KPipeline(lang_code='b')

    async def tts_stream(full_text: str):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]
        for sent in sentences:
            for gs, ps, audio in pipeline(sent, voice='af_heart'):
                sd.play(audio, 24000, blocking=False)
                await asyncio.sleep(0)

    async def hear():
        last = ""
        while True:
            heard = recorder.text()
            if heard and heard.strip() and heard != last:
                print("User:", heard)
                last = heard
                try:
                    answer = await rag_chain.ainvoke(heard)
                    print("Answer:", answer)
                    await tts_stream(answer)
                except Exception as e:
                    print("RAG error:", e)
            await asyncio.sleep(0.15)

    asyncio.run(hear())

if __name__ == "__main__":
    main()
