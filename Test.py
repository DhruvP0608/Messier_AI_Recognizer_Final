from google import genai

# If GOOGLE_API_KEY is set in Windows Environment Variables, 
# the client finds it automatically.
client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview", 
    contents="When is Ascension 2026?"
)

print(response.text)