from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 
import websockets
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import VectorizeDb


retriever = VectorizeDb.vector_store.as_retriever()

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", disable_streaming = False, api_key = api_key)

template = """You are a smart receptionist for AI Department of St Josephs College of Engineering and Technology, You were created by Aibel Bin Zacariah, student of Semester 5, Artificial Intelligence. 
The context will be given which contains details about every program happening in Artificial Intelligence, Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

async def SttConnection():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri, open_timeout = 10) as websocket: 
        print("Connected")
        async for message in websocket:
            if message: 
                response = await rag_chain.ainvoke(message) 
            print(response)
            return response
        
asyncio.run(SttConnection())


