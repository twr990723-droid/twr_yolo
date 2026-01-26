"""
Google Sheet 記錄模組
用於將違規記錄寫入 Google Sheet
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

def init_google_sheet(sheet_name="違規記錄"):
    """
    初始化 Google Sheet 連接
    
    Args:
        sheet_name: Google Sheet 的名稱
        
    Returns:
        worksheet: Google Sheet worksheet 物件
    """
    try:
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_file(
            "credentials.json",
            scopes=scope
        )
        
        gc = gspread.authorize(creds)
        sheet = gc.open(sheet_name)
        worksheet = sheet.sheet1
        
        # 檢查是否有表頭，如果沒有則添加
        try:
            headers = worksheet.row_values(1)
            if not headers or headers != ["日期", "學號", "姓名", "違規內容記錄"]:
                worksheet.insert_row(["日期", "學號", "姓名", "違規內容記錄"], 1)
        except:
            worksheet.insert_row(["日期", "學號", "姓名", "違規內容記錄"], 1)
        
        return worksheet
    except Exception as e:
        print(f"Google Sheet 初始化失敗: {str(e)}")
        return None


def log_violation(worksheet, student_id, name, violations):
    """
    記錄違規資料到 Google Sheet
    
    Args:
        worksheet: Google Sheet worksheet 物件
        student_id: 學號
        name: 姓名
        violations: 違規物品列表 (list of dict with 'class_name' and 'confidence')
    
    Returns:
        bool: 是否記錄成功
    """
    if worksheet is None:
        return False
    
    try:
        # 格式化日期
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 格式化違規內容
        # 當辨識類別為0時顯示為合規，其他類別顯示為不合規
        violations.sort(key=lambda x: x['confidence'], reverse=True)
        violation_str = "合規" if violations[0]['class_name'] == '0' else "不合規"
        
        # violation_items = []
        # for v in violations:
        #     # 假設 class_id 或 class_name 為 0 或 "0" 時表示合規
        #     if v.get('class_id') == 0 or v.get('class_name') == '0':
        #         violation_items.append("合規")
        #     else:
        #         violation_items.append(f"不合規: {v['class_name']}({v['confidence']:.1%})")
        
        # violation_str = ", ".join(violation_items)
        
        # 添加記錄
        row = [date_str, student_id, name, violation_str]
        worksheet.append_row(row)
        
        print(f"[Google Sheet] 記錄成功: {student_id} - {violation_str}")
        return True
    except Exception as e:
        print(f"[Google Sheet] 記錄失敗: {str(e)}")
        return False
