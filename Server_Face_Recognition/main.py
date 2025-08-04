import os
from typing import Union
import asyncio, json
from fastapi import FastAPI, HTTPException, status, Form, File, UploadFile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from Core.sock import ConnectionManager
from Core.face_recognition import FaceRecognitionSystem
from Core.information import InformationSystem
import cv2
from pynput import keyboard
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from typing import Generator, List, Dict, Any
from queue import Queue
import threading
from dataclasses import dataclass
from typing import List, Dict, Any
import time
import shutil  # thêm import cho xóa folder
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()

app.mount(
     "/static",
     StaticFiles(directory=os.path.join(BASE_DIR, "static")),
     name="static"
)

app.mount(
     "/Data",
     StaticFiles(directory=os.path.join(BASE_DIR, "Data")),
     name="media"
)
#Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu trúc dữ liệu để lưu frame và metadata
@dataclass
class FrameData:
    frame: np.ndarray
    annotated_frame: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    
    
    

UPLOAD_DIR = "Data/video"
# Khai báo biến toàn cục ở đây
video_processor = None  # Khai báo global đối tượng VideoProcessor
stop_program = False
manager = ConnectionManager()

@app.get("/profileblacklist/")
async def process_data():
    jf = open("Data/csdl.json")
    data = json.load(jf)
    jf.close()
    return {"data": data, "status": status.HTTP_200_OK}



# Endpoint nhận thông tin từ client
@app.post("/facialrecognition/")
async def process_data(camera: bool = Form(...), video: Union[UploadFile, None] = File(None)):
    global video_processor
    # Nếu file được gửi mà filename rỗng thì coi như không có file
    if video is not None and video.filename == "":
        video = None

    if camera is True and video is None:
        # Sử dụng webcam (ví dụ: camera index 0)
        video_processor = VideoProcessor(camera_source="Data/video/huy.mp4")
        return {"data": {"path": 2}, "status": status.HTTP_200_OK}
    elif camera is True and video is not None:
        return {"error": "chỉ được chọn camera hoặc video", "status": status.HTTP_400_BAD_REQUEST}
    elif camera is False and video is not None:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, video.filename)
        # Nếu file đã tồn tại thì sử dụng luôn
        if not os.path.exists(file_path):
            contents = await video.read()
            with open(file_path, "wb") as f:
                f.write(contents)
        # Sử dụng file video đã lưu làm nguồn
        video_processor = VideoProcessor(camera_source="Data/video/huy.mp4")
        return {"data": {"path": file_path}, "status": status.HTTP_200_OK}
    elif camera is False and video is None:
        return {"error": "hãy chọn camera hoặc video", "status": status.HTTP_400_BAD_REQUEST}
    else:
        raise HTTPException(status_code=400, detail="Dữ liệu không hợp lệ.")


def break_loop(key):
    global stop_program
    try:
        if key == keyboard.Key.esc:
            stop_program = True
            if video_processor is not None:
                video_processor.stop()  # Dừng VideoProcessor khi ESC được nhấn
                print("Đã dừng VideoProcessor")
    except AttributeError:
        pass
    
    
class VideoProcessor:
    def __init__(self, camera_source):
        self.camera = cv2.VideoCapture(camera_source)
        self.frame_data = None
        self.lock = threading.Lock()
        self.running = True
        
        # Khởi tạo instance của FaceRecognitionSystem
        self.face_recognition = FaceRecognitionSystem()
        self.information_system = InformationSystem()
        
        # Khởi động thread xử lý
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()

        # Đường dẫn đến folder database
        database_path = "Dataset"
        embeddings_path = "embeddings.pkl"
        # Thử tải embeddings có sẵn từ đối tượng của lớp FaceRecognitionSystem
        if not self.face_recognition.load_embeddings(embeddings_path):
            # Nếu chưa có embeddings thì xử lý database folder
            self.face_recognition.process_database_folder(database_path)
            # Lưu embeddings để lần sau dùng lại
            self.face_recognition.save_embeddings(embeddings_path)

        

    def _process_frames(self):
        while self.running:
            success, frame = self.camera.read()
            if not success:
                continue
            
            metadata = {}
            data = []
           
            results, result_img = self.face_recognition.process_image(frame)
            # In kết quả
            for i, result in enumerate(results):
                print(f"Face {i+1}: {result['name']} (confidence: {result['confidence']:.2f})")
                
                profile = self.information_system.information_retrieval(result['name'])
                data.append(profile)
            
            # Tạo metadata
            metadata["detections"]  = data
            metadata["timestamp"] = time.time()
            #cv2.imshow("Face Recognition", result_img)
            #cv2.waitKey(1)
            # Cập nhật frame_data với lock để thread-safe
            with self.lock:
                self.frame_data = FrameData(
                    frame=frame,
                    annotated_frame=result_img,
                    metadata=metadata,
                    timestamp=time.time()
                )

            # Đợi một khoảng thời gian nhỏ để giảm tải CPU
            time.sleep(0.009)

    def get_current_frame_data(self) -> FrameData:
        with self.lock:
            return self.frame_data

    def stop(self):
        """Phương thức dừng VideoProcessor: giải phóng camera và dừng thread."""
        self.running = False
        self.camera.release()
        if self.process_thread.is_alive():
            self.process_thread.join()

    def __del__(self):
        self.stop()


async def generate_frames() -> Generator:
    while True:
        frame_data = video_processor.get_current_frame_data()
        if frame_data is None:
            await asyncio.sleep(0.009)
            continue

        # Chuyển frame sang định dạng bytes
        _, buffer = cv2.imencode('.jpg', frame_data.annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.009)

@app.get("/video-stream")
async def video_stream():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/supervision_streaming/")
async def websocket_endpoint(websocket: WebSocket):
    connection_id = await manager.connect(websocket)
    
    global stop_program
    stop_program = False
    listener = keyboard.Listener(on_press=break_loop)
    listener.start()

    async def receiver():
        try:
            while True:
                data = await websocket.receive_text()
                print("Received message from client:", data)
                request = json.loads(data)
                print("Request data:", request)
                path = request.get("path", "")
                action = request.get("action", "")
                print("Path:", path)
                print("SUPERVISE STARTED")
                
                if action == "off":
                    global stop_program
                    stop_program = True
                    if video_processor is not None:
                        video_processor.stop()  # Dừng VideoProcessor khi nhận lệnh off
                        print("Đã dừng VideoProcessor")
                    try:
                        await manager.send_personal_message(
                            {"data_state": "supervise_complete"},
                            connection_id
                        )
                    except Exception as e:
                        print("Send error in receiver:", e)
                    listener.stop()
                    manager.disconnect(connection_id)
                    break  # Thoát vòng lặp receiver sau khi xử lý off
        except WebSocketDisconnect:
            manager.disconnect(connection_id)
    
    # Chạy receiver song song với vòng lặp chính
    receiver_task = asyncio.create_task(receiver())
    
    last_sent_timestamp = 0
    try:
        while True:
            if stop_program:
                break

            if video_processor is None:
                await asyncio.sleep(0.009)
                continue

            frame_data = video_processor.get_current_frame_data()
            if frame_data is None:
                await asyncio.sleep(0.009)
                continue

            if frame_data.timestamp > last_sent_timestamp:
                try:
                    await manager.send_personal_message(
                        {"profile": frame_data.metadata},
                        connection_id
                    )
                except Exception as e:
                    print("Send error in main loop:", e)
                    break  # Nếu lỗi xảy ra do kết nối đã đóng
                last_sent_timestamp = frame_data.timestamp

            await asyncio.sleep(0.009)
            
        print("SUPERVISE COMPLETE")
        # Gửi thông báo hoàn thành nếu cần (chỉ khi kết nối chưa đóng)
        try:
            await manager.send_personal_message(
                {"data_state": "supervise_complete"},
                connection_id
            )
        except Exception as e:
            print("Final send error:", e)
        listener.stop()
        manager.disconnect(connection_id)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
        receiver_task.cancel()
        
        
from typing import List

@app.post("/create_data/")
async def create_data(
    avata: UploadFile = File(...),
    name: str = Form(...),
    full_name: str = Form(...),
    year_of_birth: str = Form(...),
    hometown: str = Form(...),
    id_number: str = Form(...),
    violation: str = Form(...),
    data: List[UploadFile] = File(...)
):
    # Lưu file avata vào Data/avatar
    avatar_folder = "Data/avatar"
    os.makedirs(avatar_folder, exist_ok=True)
    avatar_path = os.path.join(avatar_folder, avata.filename)
    contents = await avata.read()
    with open(avatar_path, "wb") as f:
        f.write(contents)
    
    # Tạo folder lưu dữ liệu ảnh theo trường name
    data_folder = os.path.join("Dataset", name)
    os.makedirs(data_folder, exist_ok=True)
    data_paths = []
    for file in data:
        file_path = os.path.join(data_folder, file.filename)
        file_contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_contents)
        data_paths.append(file_path)
    
    # Cập nhật file JSON (Data/csdl.json)
    csdl_file = "Data/csdl.json"
    if os.path.exists(csdl_file):
        with open(csdl_file, "r", encoding="utf-8") as jf:
            try:
                csdl_data = json.load(jf)
            except Exception:
                csdl_data = {}
    else:
        csdl_data = {}
    
    csdl_data[name] = {
        "avata": avatar_path,
        "full_name": full_name,
        "year_of_birth": year_of_birth,
        "hometown": hometown,
        "id_number": id_number,
        "violation": violation,
        "data": data_paths
    }
    
    # Lưu file JSON với định dạng UTF-8
    with open(csdl_file, "w", encoding="utf-8") as jf:
        json.dump(csdl_data, jf, ensure_ascii=False, indent=4)
    
    return {"message": "Thêm dữ liệu thành công.", "data": csdl_data[name], "status": status.HTTP_200_OK}


from typing import List, Union, Optional
from fastapi import HTTPException

@app.patch("/update_data/")
async def update_data(
    name: str = Form(...),  # Dùng để tìm record cần cập nhật (không cho phép update)
    avata: Optional[UploadFile] = File(None),
    full_name: Optional[str] = Form(None),
    year_of_birth: Optional[str] = Form(None),
    hometown: Optional[str] = Form(None),
    id_number: Optional[str] = Form(None),
    violation: Optional[str] = Form(None),
    data: Optional[List[UploadFile]] = File(None)
):
    csdl_file = "Data/csdl.json"
    # Đọc file JSON với định dạng UTF-8
    if os.path.exists(csdl_file):
        with open(csdl_file, "r", encoding="utf-8") as jf:
            try:
                csdl_data = json.load(jf)
            except Exception:
                csdl_data = {}
    else:
        csdl_data = {}

    # Kiểm tra xem có tồn tại đối tượng với key là name không
    if name not in csdl_data:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu với name: {name}")

    # Lấy record cần update
    record = csdl_data[name]

    # Nếu avata được cung cấp và có filename hợp lệ, lưu file vào folder "Data/avatar" và update
    if avata is not None and avata.filename.strip() != "":
        avatar_folder = "Data/avatar"
        os.makedirs(avatar_folder, exist_ok=True)
        avatar_path = os.path.join(avatar_folder, avata.filename)
        contents = await avata.read()
        with open(avatar_path, "wb") as f:
            f.write(contents)
        record["avata"] = avatar_path

    # Cập nhật các trường text chỉ khi có giá trị không rỗng (sau khi loại khoảng trắng)
    if full_name is not None and full_name.strip() != "":
        record["full_name"] = full_name
    if year_of_birth is not None and year_of_birth.strip() != "":
        record["year_of_birth"] = year_of_birth
    if hometown is not None and hometown.strip() != "":
        record["hometown"] = hometown
    if id_number is not None and id_number.strip() != "":
        record["id_number"] = id_number
    if violation is not None and violation.strip() != "":
        record["violation"] = violation

    # Nếu trường data (nhiều file ảnh) được cung cấp, lưu file vào folder "Dataset/"+name và update list file đường dẫn
    if data is not None and len(data) > 0:
        data_folder = os.path.join("Dataset", name)
        os.makedirs(data_folder, exist_ok=True)
        data_paths = []
        for file in data:
            # Kiểm tra nếu file có filename hợp lệ
            if file.filename.strip() == "":
                continue
            file_path = os.path.join(data_folder, file.filename)
            file_contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_contents)
            data_paths.append(file_path)
        # Chỉ cập nhật nếu có file hợp lệ được upload
        if data_paths:
            record["data"] = data_paths

    # Cập nhật lại record trong csdl_data
    csdl_data[name] = record
    # Ghi file JSON với định dạng UTF-8
    with open(csdl_file, "w", encoding="utf-8") as jf:
        json.dump(csdl_data, jf, ensure_ascii=False, indent=4)

    return {
        "message": "Cập nhật dữ liệu thành công.",
        "data": csdl_data[name],
        "status": status.HTTP_200_OK
    }

@app.delete("/delete_data/")
async def delete_data(name: str = Form(...)):
    csdl_file = "Data/csdl.json"
    # Đọc file JSON với định dạng UTF-8
    if os.path.exists(csdl_file):
        with open(csdl_file, "r", encoding="utf-8") as jf:
            try:
                csdl_data = json.load(jf)
            except Exception:
                csdl_data = {}
    else:
        csdl_data = {}

    # Kiểm tra record có tồn tại với key là name không
    if name not in csdl_data:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy dữ liệu với name: {name}")

    record = csdl_data[name]

    # Xóa file avata nếu tồn tại
    if "avata" in record:
        avatar_path = record["avata"]
        if os.path.exists(avatar_path) and os.path.isfile(avatar_path):
            os.remove(avatar_path)

    # Xóa folder chứa dữ liệu ảnh: "Dataset/"+name
    data_folder = os.path.join("Dataset", name)
    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        shutil.rmtree(data_folder)

    # Xóa record khỏi csdl_data
    del csdl_data[name]

    # Ghi lại file JSON với định dạng UTF-8
    with open(csdl_file, "w", encoding="utf-8") as jf:
        json.dump(csdl_data, jf, ensure_ascii=False, indent=4)

    return {"message": "Xoá dữ liệu thành công.", "status": status.HTTP_200_OK}

@app.post("/update_model/")
async def update_data():
     # Khởi tạo instance của FaceRecognitionSystem
    face_recognition = FaceRecognitionSystem()
    
    # Đường dẫn đến folder database
    database_path = "Dataset"
    embeddings_path = "embeddings.pkl"
    # Nếu chưa có embeddings thì xử lý database folder
    face_recognition.process_database_folder(database_path)
    # Lưu embeddings để lần sau dùng lại
    face_recognition.save_embeddings(embeddings_path)
    return {"message": "Cập nhật dữ liệu model thành công.", "status": status.HTTP_200_OK}
