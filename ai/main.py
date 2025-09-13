from groq import Groq
from dotenv import load_dotenv
import os 
import websockets
import asyncio


async def SttConnection():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket: 
        print("Connected")
        async for message in websocket: 
            print(message)


asyncio.run(SttConnection())
load_dotenv()
api_key = os.get("GROQ_API_KEY")

client = Groq(api_key)

def callAssistant(message):
    response = client.chat.completions.create(
        messages = [
            {"role": 'f{message}'}     
        ],
        model = "llama-3.3-70b-versatile"
    )
    return (response[0].message.content)