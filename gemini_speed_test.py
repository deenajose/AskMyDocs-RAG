import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(
    api_key=api_key
)

print("Testing Gemini...")

start = time.time()

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Answer in one short sentence: What is machine learning?"
)

elapsed = time.time() - start

print("\nResponse:")
print(response.text)

print(f"\nGemini took: {elapsed:.2f} seconds")