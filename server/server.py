from fastapi import FastAPI, WebSocket
from RealtimeSTT import AudioToTextRecorder
from starlette.websockets import WebSocketDisconnect
import asyncio

app = FastAPI()
recorder = AudioToTextRecorder()


async def backgroundrun():
    while True:

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, 
                                   recorder.text)
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def startup():
    asyncio.create_task(backgroundrun())


@app.websocket('/ws')
async def websocket(websocket: WebSocket):
    
    await websocket.accept()
    host = websocket.client.host
    print(f"{host} connected")
    

    try:
        while True:
            latest_transcript = recorder.text()
            await websocket.send_text(latest_transcript)
            await asyncio.sleep(0.1) 

    
    except WebSocketDisconnect:
        recorder.stop()
        print(f"Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 
        
        

