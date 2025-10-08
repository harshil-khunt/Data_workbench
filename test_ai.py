import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load the environment variables from your .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create an instance of the Gemini Pro model
# NEW CODE
# NEW, CORRECT CODE
model = genai.GenerativeModel('models/gemini-2.5-pro')
# Send a simple prompt
response = model.generate_content("In one sentence, what is a data scientist?")

# Print the response from the model
print(response.text)