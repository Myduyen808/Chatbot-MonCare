from google import genai

client = genai.Client(api_key="AIzaSyBKSqZmIYioKP909gta2rHQR5oaptBAbaM")

res = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Xin chào"
)

print(res.text)