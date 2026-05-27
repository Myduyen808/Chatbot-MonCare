"""
🔍 TEST GEMINI MODELS - Tìm model nào hoạt động với API key của bạn
Chạy: python test_gemini_models.py
"""

import os
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, list_models
import google.generativeai as genai

load_dotenv(override=True)

# Kiểm tra API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ Thiếu GOOGLE_API_KEY trong .env!")
    exit(1)

print("🔑 API Key: OK")
print("="*60)

# Configure
genai.configure(api_key=API_KEY)

print("📋 DANH SÁCH TẤT CẢ MODELS CÓ THỂ DÙNG:")
print("="*60)

# 1. List tất cả models available
available_models = []
for model in list_models():
    if 'generateContent' in model.supported_generation_methods:
        model_name = model.name.split('/')[-1]
        available_models.append(model_name)
        print(f"✅ {model_name:<25} | {model.description[:50]}")

print("\n" + "="*60)
print("🧪 TEST 10 MODELS PHỔ BIẾN NHẤT:")

# 2. Test từng model phổ biến
test_models = [
    "gemini-pro",
    "gemini-1.0-pro",
    "gemini-pro-vision", 
    "gemini-1.5-flash",
    "gemini-1.5-flash-exp",
    "gemini-1.5-pro",
    "gemini-1.5-pro-exp",
    "gemini-1.0-pro-vision",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-thinking-exp",
]

working_models = []
for model_name in test_models:
    try:
        # Test native Google AI SDK
        model = GenerativeModel(model_name)
        response = model.generate_content("Test")
        print(f"✅ {model_name:<25} | OK | Tokens: {len(response.text)}")
        working_models.append(model_name)
    except Exception as e:
        print(f"❌ {model_name:<25} | {str(e)[:60]}")

print("\n" + "🏆"*20)
print("🎯 MODELS HOẠT ĐỘNG VỚI API KEY CỦA BẠN:")
for model in working_models:
    print(f"   🔸 {model}")

if working_models:
    print(f"\n💡 DÙNG MODEL NÀY TRONG RAGAS: '{working_models[0]}'")
else:
    print("❌ Không model nào hoạt động! Tạo API key mới.")

print("\n📈 Kết luận:")
print("1. Copy model đầu tiên trong danh sách trên vào get_judge_llm()")
print("2. gemini-pro thường OK nhất")
print("3. Nếu không có model nào → API key hết hạn")