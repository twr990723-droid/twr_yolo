"""
即時人臉辨識系統 (異步版本)
從攝影機讀取影像,與資料庫中的embeddings進行比對
使用多線程實現異步處理:
- 主線程: 人臉偵測 + 畫面顯示
- 工作線程: embedding提取 + 比對
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import numpy as np
import csv
import os
from PIL import Image
import threading
import queue
from collections import deque
import time

# 設定參數
DATABASE_CSV = "face_embeddings_database.csv"
SIMILARITY_THRESHOLD = 1.0  # 相似度閾值,越小越嚴格
CAMERA_INDEX = 0  # 攝影機編號,通常0是預設攝影機
QUEUE_MAX_SIZE = 3  # 隊列最大尺寸,避免累積太多待處理影像

# 顏色定義 (BGR格式)
COLOR_RECOGNIZED = (0, 255, 0)  # 綠色 - 辨識成功
COLOR_UNKNOWN = (0, 0, 255)     # 紅色 - 未知人臉
COLOR_DETECTING = (0, 165, 255)  # 橘色 - 偵測中
COLOR_TEXT = (255, 255, 255)    # 白色 - 文字


class FaceRecognitionSystem:
    """人臉辨識系統類別 (異步版本)"""
    
    def __init__(self, database_csv, threshold=1.0):
        """
        初始化人臉辨識系統
        
        Args:
            database_csv: CSV資料庫檔案路徑
            threshold: 相似度閾值
        """
        self.database_csv = database_csv
        self.threshold = threshold
        self.database = []
        
        # 初始化模型
        print("正在載入模型...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 偵測用的 MTCNN (主線程使用)
        self.mtcnn_detect = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=40
        )
        
        # 提取embedding用的 MTCNN 和 Resnet (工作線程使用)
        self.mtcnn_embed = MTCNN(
            keep_all=False,
            device=self.device
        )
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
            print("✓ 使用GPU加速")
        else:
            print("✓ 使用CPU運算")
        
        # 異步處理相關
        self.face_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.result_queue = queue.Queue()  # 存放辨識結果的隊列
        self.worker_thread = None
        self.running = False
        self.next_face_id = 0
        
        # 結果顯示緩存（避免閃爍）
        self.last_result = None  # 緩存最後一次的辨識結果
        self.result_display_duration = 3.0  # 結果顯示 3 秒後過期
        
        # 載入資料庫
        self.load_database()
        
    def load_database(self):
        """從CSV檔案載入人臉資料庫"""
        if not os.path.exists(self.database_csv):
            print(f"錯誤: 找不到資料庫檔案 '{self.database_csv}'")
            print("請先執行 generate_embeddings.py 建立資料庫")
            return False
        
        print(f"正在載入資料庫: {self.database_csv}")
        
        with open(self.database_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # 跳過標題行
            
            for row in reader:
                filename = row[0]
                embedding = np.array([float(x) for x in row[1:]])
                self.database.append({
                    'filename': filename,
                    'name': os.path.splitext(filename)[0],  # 移除副檔名作為名字
                    'embedding': embedding
                })
        
        print(f"✓ 已載入 {len(self.database)} 筆人臉資料")
        return True
    
    def extract_embedding(self, face_img):
        """
        從人臉影像提取embedding
        
        Args:
            face_img: PIL Image 物件 (已經是裁切好的人臉)
            
        Returns:
            embedding: numpy array or None
        """
        try:
            # 使用 MTCNN 對人臉進行對齊和標準化
            aligned = self.mtcnn_embed(face_img)
            
            if aligned is None:
                return None
            
            # 檢查維度並調整
            if aligned.dim() == 3:
                # [3, 160, 160] -> [1, 3, 160, 160]
                aligned = aligned.unsqueeze(0)
            elif aligned.dim() == 4:
                # 已經是 [1, 3, 160, 160]，不需要處理
                pass
            else:
                return None
            
            # 移到正確的設備
            if torch.cuda.is_available():
                aligned = aligned.cuda()
            
            # 提取embedding
            with torch.no_grad():
                embedding = self.resnet(aligned).detach().cpu().numpy()[0]
            
            return embedding
            
        except Exception as e:
            print(f"提取embedding時發生錯誤: {str(e)}")
            return None
    
    def find_match(self, embedding):
        """
        在資料庫中尋找最相似的人臉
        
        Args:
            embedding: 要比對的embedding
            
        Returns:
            match_name: 匹配的名字 (如果有)
            distance: 最小距離
        """
        if len(self.database) == 0:
            return None, float('inf')
        
        min_distance = float('inf')
        match_name = None
        
        for person in self.database:
            # 計算歐氏距離
            distance = np.linalg.norm(embedding - person['embedding'])
            
            if distance < min_distance:
                min_distance = distance
                match_name = person['name']
        
        return match_name, min_distance
    
    def recognition_worker(self): #第二位同學
        """
        工作線程: 從隊列中取出人臉影像,提取embedding並比對
        """
        print("✓ 辨識工作線程已啟動")
        
        while self.running:
            try:
                # 從隊列中取出任務 (timeout避免阻塞)
                task = self.face_queue.get(timeout=0.1)
                
                face_id = task['face_id']
                face_img = task['face_img']
                box = task['box']
                
                # 提取embedding
                start_time = time.time()
                embedding = self.extract_embedding(face_img)
                process_time = time.time() - start_time
                
                if embedding is not None:
                    # 比對資料庫
                    match_name, distance = self.find_match(embedding)
                    
                    # 判斷結果: 大於閾值 = 陌生人, 小於閾值 = 當事人名稱
                    if distance > self.threshold:
                        result_name = "Stranger"
                        status = "stranger"
                    else:
                        result_name = match_name
                        status = "matched"
                    
                    # 輸出到終端機
                    print(f"[辨識完成] {result_name} (距離: {distance:.3f}, 處理時間: {process_time:.3f}s)")
                    
                    # 將結果放入結果隊列
                    result = {
                        'box': box,
                        'name': result_name,
                        'distance': distance,
                        'status': status,
                        'timestamp': time.time()
                    }
                    self.result_queue.put(result)
                    




                else:
                    print(f"[辨識失敗] 無法提取 embedding")
                
                self.face_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"工作線程錯誤: {str(e)}")
        
        print("✓ 辨識工作線程已關閉")
    
    def draw_detection_boxes(self, frame, boxes, probs):
        """
        繪製人臉偵測方框 (主線程調用)
        
        Args:
            frame: 影像畫面
            boxes: 人臉方框座標
            probs: 偵測信心度
            
        Returns:
            frame: 繪製後的畫面
            face_data: 偵測到的人臉資訊列表
        """
        
        if boxes is None:
            return frame
        
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            # 取得方框座標
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # 預設橘色方框和標籤
            color = COLOR_DETECTING
            label = "face detected"
            
            cv2.putText(frame, label, (x1 + 5, y1 - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
                # 只有偵測，尚未辨識
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        return frame
    
    
    def draw_recognition_results(self, frame):
        """
        在畫面角落繪製辨識結果
        使用緩存機制避免文字閃爍
        
        Args:
            frame: 影像畫面
            
        Returns:
            frame: 繪製後的畫面
        """
        # 如果有新結果，更新緩存
        if not self.result_queue.empty():
            self.last_result = self.result_queue.get_nowait()
        
        # 如果沒有緩存的結果，直接返回
        if self.last_result is None:
            return frame
        
        # 檢查結果是否過期
        current_time = time.time()
        if current_time - self.last_result['timestamp'] > self.result_display_duration:
            self.last_result = None  # 清除過期結果
            return frame
        
        result = self.last_result

        # 根據狀態設定顏色
        if result['status'] == 'matched':
            # 辨識成功 - 綠色
            color = COLOR_RECOGNIZED
        else:  # stranger
            # 陌生人 - 紅色
            color = COLOR_UNKNOWN
        
        label = f"{result['name']}"
        sub_label = f"distance: {result['distance']:.3f}"
        
        # 繪製半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 45), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 繪製文字
        cv2.putText(frame, label, (5, 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, sub_label, (5, 36), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)

        return frame


    
    def run(self):
        """啟動即時辨識系統 (異步版本)"""
        if len(self.database) == 0:
            print("資料庫為空,無法啟動辨識系統")
            return
        
        # 開啟攝影機
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print("錯誤: 無法開啟攝影機")
            return
        
        # 設定攝影機解析度為 640x640
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 確認實際設定的解析度
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ 攝影機解析度: {actual_width}x{actual_height}")
        
        print("\n" + "="*60)
        print("即時人臉辨識系統已啟動 (異步模式)")
        print("="*60)
        print("操作說明:")
        print("  - 按 'q' 或 'ESC' 鍵離開")
        print("  - 按 's' 鍵截圖")
        print(f"  - 相似度閾值: {self.threshold}")
        print(f"  - 資料庫人數: {len(self.database)}")
        print("="*60)
        
        # 啟動工作線程
        self.running = True
        self.worker_thread = threading.Thread(target=self.recognition_worker, daemon=True)
        self.worker_thread.start()
        
        # 創建固定大小的窗口
        window_name = 'Face Recognition System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 創建可調整大小的窗口
        cv2.resizeWindow(window_name, actual_width, actual_height)  # 設定窗口顯示大小為 1280x720
        print(f"✓ 窗口顯示大小: {actual_width}x{actual_height}")
        
        frame_count = 0
        detect_every_n_frames = 1 # 每5幀偵測一次人臉
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("無法讀取攝影機畫面")
                    break
                
                frame_count += 1
                
                # 每N幀才進行人臉偵測
                if frame_count % detect_every_n_frames == 0:
                    # 轉換為RGB (OpenCV使用BGR)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    
                    # 偵測人臉
                    faces, probs = self.mtcnn_detect.detect(pil_img)
                    
                    # 將偵測到的人臉加入處理隊列
                    if faces is not None:
                        for box in faces:
                            # 嘗試加入隊列 (如果隊列滿了就跳過)
                            try:
                                task = {
                                    'face_id': self.next_face_id,
                                    'face_img': pil_img,
                                    'box': box
                                }
                                self.face_queue.put_nowait(task)
                                self.next_face_id += 1
                            except queue.Full:
                                # 隊列已滿,跳過此人臉
                                pass

                        frame = self.draw_detection_boxes(frame, faces, probs)

                    
                frame = self.draw_recognition_results(frame)

                # 顯示畫面（使用命名窗口）
                cv2.imshow(window_name, frame)
                
                # 按鍵處理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    print("\n正在關閉系統...")
                    break
                elif key == ord('s'):  # 截圖
                    filename = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"✓ 截圖已儲存: {filename}")
                    
        except KeyboardInterrupt:
            print("\n接收到中斷訊號,正在關閉...")
        finally:
            # 停止工作線程
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=2.0)
            
            # 釋放資源
            cap.release()
            cv2.destroyAllWindows()
            print("✓ 系統已關閉")


def main():
    """主程式"""
    print("="*60)
    print("即時人臉辨識系統")
    print("="*60)
    
    # 建立辨識系統
    system = FaceRecognitionSystem(
        database_csv=DATABASE_CSV,
        threshold=SIMILARITY_THRESHOLD
    )
    
    # 啟動辨識
    system.run()


if __name__ == "__main__":
    main()
