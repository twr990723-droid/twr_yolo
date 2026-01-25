"""
Face Embeddings Generator
讀取face_images資料夾中的所有影像，提取人臉embeddings並存入CSV檔案
"""

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import csv
from pathlib import Path

# 設定路徑
IMAGE_FOLDER = "face_images"
OUTPUT_CSV = "face_embeddings_database.csv"
OUTPUT_IMAGE_FOLDER = "face_images_with_boxes"  # 存放標註方框的影像

# 支援的圖片格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

def initialize_models():

    """初始化MTCNN和InceptionResnet模型"""
    print("正在載入模型...")
    # 使用 keep_all=False 來只保留最可能的臉，並且我們需要同時獲取 bounding boxes
    mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')
    # 建立第二個 MTCNN 用於偵測臉部位置（用於畫框）
   
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    
    # 如果有GPU則使用GPU
    if torch.cuda.is_available():
        resnet = resnet.cuda()
        print("使用GPU加速")
    else:
        print("使用CPU運算")
    
    print("模型載入完成!")
    return mtcnn, resnet


def extract_face_embedding(image_path, mtcnn, resnet, output_folder):
    """
    從圖片中提取人臉embedding並畫出臉部方框
    
    Args:
        image_path: 圖片路徑
        mtcnn: MTCNN人臉檢測模型（用於提取embedding）
        mtcnn_detect: MTCNN人臉檢測模型（用於偵測位置）
        resnet: InceptionResnet embedding模型
        output_folder: 輸出標註影像的資料夾
    
    Returns:
        embedding: numpy array of shape (512,) or None if no face detected
    """
    try:
        # 讀取圖片
        img = Image.open(image_path).convert("RGB")
        img_copy = img.copy()  # 複製一份用於畫框
        
        # 偵測人臉位置（獲取bounding boxes）
        boxes, probs = mtcnn.detect(img)
        
        # 偵測並對齊人臉（用於提取embedding）
        img_cropped = mtcnn(img)
        
        if img_cropped is None or boxes is None:
            print(f"  ⚠ 警告: {os.path.basename(image_path)} 未偵測到人臉")
            return None
        
        # 在圖片上畫出臉部方框
        draw = ImageDraw.Draw(img_copy)
        for box, prob in zip(boxes, probs):
            # box 格式: [x1, y1, x2, y2]
            draw.rectangle(box.tolist(), outline='red', width=3)
            # 也可以顯示信心度
            draw.text((box[0], box[1]-10), f'{prob:.2f}', fill='red')
        
        # 儲存標註後的影像
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        img_copy.save(output_path)
        
        # 增加batch dimension
        img_cropped = img_cropped.unsqueeze(0)
        
        # 如果有GPU則將資料移到GPU
        if torch.cuda.is_available():
            img_cropped = img_cropped.cuda()
        
        # 提取embedding
        with torch.no_grad():
            embedding = resnet(img_cropped).detach().cpu().numpy()[0]
        
        return embedding
        
    except Exception as e:
        print(f"  ✗ 錯誤: 處理 {os.path.basename(image_path)} 時發生錯誤: {str(e)}")
        return None


def process_images_to_csv():
    """
    處理face_images資料夾中的所有影像
    提取embeddings並寫入CSV檔案，同時生成標註方框的影像
    """
    # 檢查資料夾是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"錯誤: 資料夾 '{IMAGE_FOLDER}' 不存在!")
        print(f"請先建立資料夾並放入影像檔案")
        return
    
    # 建立輸出資料夾（如果不存在）
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
        print(f"✓ 已建立資料夾: {OUTPUT_IMAGE_FOLDER}")
    
    # 取得所有圖片檔案
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]
    
    if len(image_files) == 0:
        print(f"警告: 資料夾 '{IMAGE_FOLDER}' 中沒有找到任何圖片檔案")
        print(f"支援的格式: {', '.join(SUPPORTED_FORMATS)}")
        return
    
    print(f"\n找到 {len(image_files)} 個圖片檔案")
    print("="*60)
    
    # 初始化模型
    mtcnn, resnet = initialize_models()
    
    # 準備CSV檔案
    csv_data = []
    successful_count = 0
    failed_count = 0
    
    # 處理每張圖片
    print("\n開始處理圖片...")
    print("="*60)
    
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(IMAGE_FOLDER, image_file)
        print(f"[{idx}/{len(image_files)}] 處理: {image_file}")
        
        # 提取embedding並畫出方框
        embedding = extract_face_embedding(image_path, mtcnn, resnet, OUTPUT_IMAGE_FOLDER)
        
        if embedding is not None:
            # 將embedding轉換為list並加入檔名
            row = [image_file] + embedding.tolist()
            csv_data.append(row)
            successful_count += 1
            print(f"  ✓ 成功提取 embedding (shape: {embedding.shape})")
            print(f"  ✓ 已儲存標註影像至: {OUTPUT_IMAGE_FOLDER}")
        else:
            failed_count += 1
    
    # 寫入CSV檔案
    if csv_data:
        print("\n" + "="*60)
        print(f"正在寫入CSV檔案: {OUTPUT_CSV}")
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 寫入標題行
            header = ['filename'] + [f'embedding_{i}' for i in range(512)]
            writer.writerow(header)
            
            # 寫入資料
            writer.writerows(csv_data)
        
        print(f"✓ CSV檔案建立成功!")
        print("="*60)
        print(f"\n處理完成統計:")
        print(f"  成功: {successful_count} 張")
        print(f"  失敗: {failed_count} 張")
        print(f"  總計: {len(image_files)} 張")
        print(f"\n輸出檔案:")
        print(f"  資料庫檔案: {OUTPUT_CSV}")
        print(f"  標註影像資料夾: {OUTPUT_IMAGE_FOLDER}")
        print(f"  Embedding維度: 512")
    else:
        print("\n錯誤: 沒有成功處理任何圖片")


if __name__ == "__main__":
    print("="*60)
    print("Face Embeddings Generator")
    print("="*60)
    process_images_to_csv()
    print("\n程式執行完畢!")
