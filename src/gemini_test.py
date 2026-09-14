import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Get the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found")
    exit()

print("✅ API key found")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Send a test question
interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="Explain machine learning in one simple sentence."
)

print("\nGemini says:")
print(interaction.output_text)