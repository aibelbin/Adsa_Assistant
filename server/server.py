from fastapi import FastApi, WebSocket


app = FastApi()

@app.websocket('/ws')
async def websocket(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_text()
    print(data)



