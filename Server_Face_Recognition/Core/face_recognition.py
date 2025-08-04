import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from pathlib import Path
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self, model_name='buffalo_sc', det_size=(640, 640), use_gpu=True):
        """
        Khởi tạo hệ thống nhận diện khuôn mặt
        Args:
            model_name: tên model (buffalo_s = mobilenet, buffalo_l = resnet50)
            det_size: kích thước ảnh đầu vào cho detection
            use_gpu: sử dụng GPU hay không
        """
        # Chọn provider dựa vào việc có sử dụng GPU hay không
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        
        # Khởi tạo FaceAnalysis với GPU
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # Database khuôn mặt
        self.known_faces = {}
        self.threshold = 18#1.2
        
        print(f"Đã khởi tạo Face Recognition System với {'GPU' if use_gpu else 'CPU'}")
        
    def process_database_folder(self, database_path):
        """
        Xử lý toàn bộ folder database để tạo embeddings
        Cấu trúc folder:
        database/
            person1/
                image1.jpg
                image2.jpg
            person2/
                image1.jpg
        """
        database_path = Path(database_path)
        
        # Xử lý từng người trong database
        for person_folder in database_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                person_embeddings = []
                
                # Xử lý từng ảnh của người đó
                for image_path in person_folder.glob('*.[jp][pn][g]'):  # matches .jpg, .png, .jpeg
                    embedding = self.get_face_embedding(str(image_path))
                    if embedding is not None:
                        person_embeddings.append(embedding)
                
                if person_embeddings:
                    # Lưu trung bình của các embeddings
                    self.known_faces[person_name] = np.mean(person_embeddings, axis=0)
                    print(f"Đã xử lý {len(person_embeddings)} ảnh của {person_name}")
                    
        print(f"Đã tạo xong database với {len(self.known_faces)} người")

    def save_embeddings(self, save_path='embeddings.pkl'):
        """
        Lưu embeddings database ra file
        """
        with open(save_path, 'wb') as f:
            pickle.dump(self.known_faces, f)
        print(f"Đã lưu embeddings của {len(self.known_faces)} người")

    def load_embeddings(self, load_path='embeddings.pkl'):
        """
        Đọc embeddings database từ file
        """
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"Đã tải embeddings của {len(self.known_faces)} người")
            return True
        return False

    def get_face_embedding(self, image):
        """
        Lấy embedding từ ảnh
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                print(f"Không tìm thấy file: {image}")
                return None
            image = cv2.imread(image)
        
        if image is None:
            return None
            
        faces = self.app.get(image)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def recognize_face(self, face_embedding):
        """
        Nhận diện một khuôn mặt dựa trên embedding
        """
        if len(self.known_faces) == 0:
            print("Chưa có database")
            return "Unknown", 0

        min_dist = float('inf')
        best_match = "Unknown"

        for name, emb in self.known_faces.items():
            dist = np.linalg.norm(face_embedding - emb)
            if dist < min_dist:
                min_dist = dist
                best_match = name

        confidence = 1 - (min_dist/self.threshold) if min_dist < self.threshold else 0
        
        print("min_dist : ", min_dist)
        if min_dist > self.threshold:
            best_match = "Unknown"
            
        return best_match, confidence

    def process_image(self, image, draw=True):
        """
        Xử lý một ảnh: phát hiện và nhận diện các khuôn mặt
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            return [], None
            
        # Copy ảnh để vẽ kết quả
        result_img = image.copy() if draw else None
        
        # Phát hiện khuôn mặt
        faces = self.app.get(image)
        
        # Kết quả nhận diện
        results = []
        
        # Xử lý từng khuôn mặt
        for face in faces:
            # Lấy embedding
            embedding = face.embedding
            
            # Nhận diện
            name, confidence = self.recognize_face(embedding)
            
            # Lưu kết quả
            result = {
                'name': name,
                'confidence': confidence,
                'bbox': face.bbox.astype(int),
                'embedding': embedding
            }
            results.append(result)
            
            # Vẽ kết quả nếu cần
            if draw and name != "Unknown":
                bbox = result['bbox']
                cv2.rectangle(result_img, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                cv2.putText(result_img, 
                          f"{name} ", #({confidence:.2f})
                          (bbox[0], bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                          (0, 255, 0), 2)
                
        return results, result_img

def main():
    # Khởi tạo hệ thống
    face_system = FaceRecognitionSystem()
    
    # Đường dẫn đến folder database
    database_path = "Dataset"
    embeddings_path = "embeddings.pkl"
    
    # Thử tải embeddings có sẵn
    if not face_system.load_embeddings(embeddings_path):
        # Nếu chưa có embeddings, xử lý database folder
        face_system.process_database_folder(database_path)
        # Lưu embeddings để lần sau dùng lại
        face_system.save_embeddings(embeddings_path)
        
    print("face_system : ", face_system)
    cap = cv2.VideoCapture(0) # for webcam access
    # to run on a video file:
    # cap = cv2.VideoCapture("path_to_video.mp4")
    if not cap.isOpened():
        print("Unable to open camera.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
    
        # Test nhận diện với ảnh mới
        #test_image = "test.jpg"
        results, result_img = face_system.process_image(frame)
        
        # In kết quả
        for i, result in enumerate(results):
            print(f"Face {i+1}: {result['name']} (confidence: {result['confidence']:.2f})")
        
        cv2.imshow("Face Recognition", result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()  
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()