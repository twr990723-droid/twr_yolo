import gspread
from google.oauth2.service_account import Credentials

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    "credentials.json",
    scopes=scope
)

gc = gspread.authorize(creds)


sheet = gc.open("測試")
worksheet = sheet.sheet1 
# 取得所有資料
all_data = worksheet.get_all_values()
print(all_data)

worksheet.append_row(["姓名", "年齡", "成績"])
