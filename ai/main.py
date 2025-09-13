from groq import Groq
from dotenv import load_dotenv
import os 

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