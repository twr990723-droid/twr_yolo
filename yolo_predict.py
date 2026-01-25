import cv2
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm

def get_user_choice():
    """獲取用戶選擇"""
    print("\n" + "="*50)
    print("YOLO 偵測系統")
    print("="*50)
    
    # 選擇模式
    print("\n請選擇模式：")
    print("1. 物件偵測 (Object Detection)")
    print("2. 實例分割 (Instance Segmentation)")
    mode_choice = input("請輸入選項 (1/2): ").strip()
    
    # 選擇輸入來源
    print("\n請選擇輸入來源：")
    print("1. 鏡頭 (Webcam)")
    print("2. 資料夾影像 (Image Folder)")
    source_choice = input("請輸入選項 (1/2): ").strip()
    
    return mode_choice, source_choice

def process_webcam(model, mode_name):
    """處理鏡頭輸入"""
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Error: 無法開啟攝像頭"
    
    # 設置攝像頭解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    print(f"\n{mode_name} - 鏡頭模式")
    print("按 'q' 鍵退出程式...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("無法讀取攝像頭畫面")
            break
        
        # 進行偵測/分割
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # 顯示結果
        cv2.imshow(f"YOLO - {mode_name}", annotated_frame)
        
        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("退出程式...")
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_images(model, mode_name):
    """處理資料夾影像 - 批次處理並儲存結果"""
    folder_path = input("\n請輸入資料夾路徑: ").strip()
    
    # 移除路徑前後的引號（如果有的話）
    folder_path = folder_path.strip('"').strip("'")
    
    if not os.path.exists(folder_path):
        print(f"錯誤：資料夾 '{folder_path}' 不存在")
        return
    
    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"錯誤：在 '{folder_path}' 中找不到圖片")
        return
    
    # 建立輸出資料夾（在來源資料夾的同階層）
    folder_name = os.path.basename(os.path.normpath(folder_path))
    parent_dir = os.path.dirname(os.path.normpath(folder_path))
    output_folder = os.path.join(parent_dir, f"{folder_name}_detected")
    
    # 如果輸出資料夾已存在，詢問是否覆蓋
    if os.path.exists(output_folder):
        print(f"\n輸出資料夾 '{output_folder}' 已存在")
        overwrite = input("是否覆蓋? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("取消處理")
            return
    else:
        os.makedirs(output_folder)
    
    print(f"\n找到 {len(image_files)} 張圖片")
    print(f"輸出資料夾: {output_folder}")
    print(f"{mode_name} - 批次處理中...")
    
    # 批次處理所有圖片
    success_count = 0
    fail_count = 0
    
    # 使用 tqdm 顯示進度條
    for img_path in tqdm(image_files, desc="處理進度", unit="張", ncols=100):
        img_name = os.path.basename(img_path)
        
        # 讀取圖片
        frame = cv2.imread(img_path)
        
        if frame is None:
            fail_count += 1
            continue
        
        try:
            # 進行偵測/分割
            results = model(frame)
            annotated_frame = results[0].plot()
            
            # 儲存結果
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, annotated_frame)
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
    
    # 顯示處理結果摘要
    print("\n" + "="*50)
    print("處理完成！")
    print("="*50)
    print(f"總共: {len(image_files)} 張")
    print(f"成功: {success_count} 張 ✅")
    print(f"失敗: {fail_count} 張 ❌")
    print(f"輸出資料夾: {output_folder}")
    print("="*50)

def main():
    # 獲取用戶選擇
    mode_choice, source_choice = get_user_choice()
    model_path = "./runs/detect/train/weights/best.pt"
    #model_path = "yolo26n.onnx"
    # model_path = "yolo11n-seg.pt"

    # 根據選擇載入模型
    if mode_choice == '1':
        mode_name = "物件偵測"
        print(f"\n載入模型: {model_path}")
    elif mode_choice == '2':
        mode_name = "實例分割"
        print(f"\n載入模型: {model_path}")
    else:
        print("無效的選擇，使用預設：物件偵測")
        model_path = "yolo11n.pt"
        mode_name = "物件偵測"
    
    # 載入模型
    model = YOLO(model_path)
    print("模型載入完成！")
    
    # 根據選擇處理輸入來源
    if source_choice == '1':
        process_webcam(model, mode_name)
    elif source_choice == '2':
        process_images(model, mode_name)
    else:
        print("無效的選擇，使用預設：鏡頭")
        process_webcam(model, mode_name)

if __name__ == "__main__":
    main()