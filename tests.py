import google.generativeai as genai
from pydantic import Field
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
genai_model = genai.GenerativeModel("gemini-2.5-flash")
response = genai_model.generate_content("what is the current fashion trend in term of color fashion")
print(response.text)
