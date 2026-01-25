import os
from ultralytics import YOLO

def get_training_config():
    """獲取訓練配置"""
    print("\n" + "="*60)
    print("YOLO 模型訓練系統")
    print("="*60)
    
    # 選擇訓練模式
    print("\n請選擇訓練模式：")
    print("1. 物件偵測 (Object Detection)")
    print("2. 實例分割 (Instance Segmentation)")
    mode_choice = input("請輸入選項 (1/2): ").strip()
    
    # 選擇預訓練模型
    if mode_choice == '1':
        print("\n物件偵測模型選擇：")
        print("1. yolo11n.pt (最快，精度較低)")
        print("2. yolo11s.pt (平衡)")
        print("3. yolo11m.pt (中等，精度較高)")
        print("4. yolo11l.pt (較慢，精度高)")
        print("5. yolo11x.pt (最慢，精度最高)")
        print("6. 自訂模型路徑")
        model_choice = input("請輸入選項 (1-6): ").strip()
        
        model_map = {
            '1': 'yolo11n.pt',
            '2': 'yolo11s.pt',
            '3': 'yolo11m.pt',
            '4': 'yolo11l.pt',
            '5': 'yolo11x.pt'
        }
        
        if model_choice in model_map:
            model_path = model_map[model_choice]
        elif model_choice == '6':
            model_path = input("請輸入模型路徑: ").strip().strip('"').strip("'")
        else:
            print("無效選擇，使用預設: yolo11n.pt")
            model_path = 'yolo11n.pt'
            
    elif mode_choice == '2':
        print("\n實例分割模型選擇：")
        print("1. yolo11n-seg.pt (最快，精度較低)")
        print("2. yolo11s-seg.pt (平衡)")
        print("3. yolo11m-seg.pt (中等，精度較高)")
        print("4. yolo11l-seg.pt (較慢，精度高)")
        print("5. yolo11x-seg.pt (最慢，精度最高)")
        print("6. 自訂模型路徑")
        model_choice = input("請輸入選項 (1-6): ").strip()
        
        model_map = {
            '1': 'yolo11n-seg.pt',
            '2': 'yolo11s-seg.pt',
            '3': 'yolo11m-seg.pt',
            '4': 'yolo11l-seg.pt',
            '5': 'yolo11x-seg.pt'
        }
        
        if model_choice in model_map:
            model_path = model_map[model_choice]
        elif model_choice == '6':
            model_path = input("請輸入模型路徑: ").strip().strip('"').strip("'")
        else:
            print("無效選擇，使用預設: yolo11n-seg.pt")
            model_path = 'yolo11n-seg.pt'
    else:
        print("無效選擇，使用預設: 物件偵測")
        model_path = 'yolo11n.pt'
    
    # 資料集路徑
    print("\n" + "-"*60)
    print("資料集配置")
    print("-"*60)
    print("請輸入 data.yaml 檔案路徑或包含 data.yaml 的資料夾路徑")
    print("範例: datasets/my_dataset 或 datasets/my_dataset/data.yaml")
    data_path = input("資料集路徑: ").strip().strip('"').strip("'")
    
    # 如果輸入的是資料夾，自動尋找 data.yaml
    if os.path.isdir(data_path):
        yaml_path = os.path.join(data_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            print(f"警告：在 {data_path} 中找不到 data.yaml")
            print("請確認資料夾中包含 data.yaml 檔案")
            return None
        data_path = yaml_path
    elif not os.path.exists(data_path):
        print(f"錯誤：找不到檔案 {data_path}")
        return None
    
    # 訓練參數
    print("\n" + "-"*60)
    print("訓練參數設定")
    print("-"*60)
    
    epochs = input("訓練輪數 epochs (預設: 100): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 100
    
    imgsz = input("圖片大小 imgsz (預設: 640): ").strip()
    imgsz = int(imgsz) if imgsz.isdigit() else 640
    
    batch = input("批次大小 batch (預設: -1 自動): ").strip()
    batch = int(batch) if batch.lstrip('-').isdigit() else -1
    
    # 進階參數（選擇性）
    print("\n是否設定進階參數？(y/n，預設: n): ", end="")
    advanced = input().strip().lower()
    
    config = {
        'model_path': model_path,
        'data_path': data_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
    }
    
    if advanced == 'y':
        lr0 = input("初始學習率 lr0 (預設: 0.01): ").strip()
        config['lr0'] = float(lr0) if lr0 else 0.01
        
        patience = input("早停耐心值 patience (預設: 50): ").strip()
        config['patience'] = int(patience) if patience.isdigit() else 50
        
        device = input("訓練設備 device (預設: 0 使用GPU，cpu 使用CPU): ").strip()
        config['device'] = device if device else '0'
    
    return config

def train_model(config):
    """執行模型訓練"""
    print("\n" + "="*60)
    print("開始訓練")
    print("="*60)
    print(f"模型: {config['model_path']}")
    print(f"資料集: {config['data_path']}")
    print(f"訓練輪數: {config['epochs']}")
    print(f"圖片大小: {config['imgsz']}")
    print(f"批次大小: {config['batch']}")
    print("="*60)
    
    # 載入模型
    print("\n載入模型...")
    model = YOLO(config['model_path'])
    
    # 準備訓練參數
    train_args = {
        'data': config['data_path'],
        'epochs': config['epochs'],
        'imgsz': config['imgsz'],
        'batch': config['batch'],
    }
    
    # 添加進階參數（如果有）
    if 'lr0' in config:
        train_args['lr0'] = config['lr0']
    if 'patience' in config:
        train_args['patience'] = config['patience']
    if 'device' in config:
        train_args['device'] = config['device']
    
    # 開始訓練
    print("\n開始訓練...\n")
    try:
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("訓練完成！")
        print("="*60)
        print(f"訓練結果已儲存至: {model.trainer.save_dir}")
        print(f"最佳模型: {model.trainer.best}")
        print("="*60)
        
        # 評估模型
        print("\n是否在驗證集上評估模型？(y/n，預設: y): ", end="")
        evaluate = input().strip().lower()
        
        if evaluate != 'n':
            print("\n正在評估模型...")
            metrics = model.val()
            print("\n評估完成！")
            
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        return False
    
    return True

def main():
    """主程式"""
    # 獲取訓練配置
    config = get_training_config()
    
    if config is None:
        print("\n配置錯誤，程式結束")
        return
    
    # 確認配置
    print("\n" + "="*60)
    print("訓練配置確認")
    print("="*60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*60)
    
    confirm = input("\n確認開始訓練？(y/n): ").strip().lower()
    
    if confirm != 'y':
        print("取消訓練")
        return
    
    # 執行訓練
    success = train_model(config)
    
    if success:
        print("\n✅ 訓練流程完成")
    else:
        print("\n❌ 訓練流程失敗")

if __name__ == "__main__":
    main()
