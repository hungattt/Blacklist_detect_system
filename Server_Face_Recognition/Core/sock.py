import json
import uuid
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        print(f"Đã kết nối WS: {connection_id}")
        return connection_id

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            self.active_connections.pop(connection_id)
            print(f"Ngắt kết nối WS: {connection_id}")

    async def send_personal_message(self, message: dict, connection_id: str):
        websocket = self.active_connections.get(connection_id)
        if websocket:
            await websocket.send_json(message)
