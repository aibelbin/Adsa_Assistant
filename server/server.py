from fastapi import FastAPI, WebSocket
from RealtimeSTT import AudioToTextRecorder
from starlette.websockets import WebSocketDisconnect
import asyncio
from ai.localgroq import sendmessage



app = FastAPI()
recorder = AudioToTextRecorder()

transcript_queue: asyncio.Queue[str] = asyncio.Queue()
response_queue : asyncio.Queue[str] = asyncio.Queue()


async def producer():
    loop = asyncio.get_running_loop()

    while True:
        latest_text = await loop.run_in_executor(None, recorder.text)
        if latest_text:
            await transcript_queue.put(latest_text)
            response = await sendmessage(message=latest_text)
            await response_queue.put(response)

        await asyncio.sleep(0.1)



    


@app.on_event("startup")
async def startup():
    asyncio.create_task(producer())

################################### for 2+ device 
# @app.websocket('/ws')
# async def websocket(websocket: WebSocket):
    
#     await websocket.accept()
#     host = websocket.client.host
#     print(f"{host} connected")
    

#     try:
#         while True:
#             transcript = await transcript_queue.get()
#             await websocket.send_text(transcript)

#     except WebSocketDisconnect:
#         recorder.stop()
#         print(f"Client disconnected.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}") 



@app.websocket('/rc')
async def websocket(websocket: WebSocket):

    await websocket.accept()
    host = websocket.client.host
    print(f"{host} connected")

    
    while True:
        to_tts = await response_queue.get()
        await websocket.send_text(to_tts)

        if WebSocketDisconnect:
            print(f"{host} disconnected")


