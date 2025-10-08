import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring API key: {e}")
    exit()

print("--- Finding available models for your API key ---")

# List all available models
for m in genai.list_models():
  # Check if the model supports the 'generateContent' method
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

print("--- End of list ---")