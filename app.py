import os
import csv
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dictionary import KNOWN_REPLIES
 
# Load the API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
 
app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)
 
class ChatMessage(BaseModel):
	message: str
 
# Load the CSV once, when the server starts, so it's fast to search later
def load_csv_data(filename="qa_data.csv"):
	data = {}
	with open(filename, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			data[row["question"].lower().strip()] = row["answer"]
	return data
 
CSV_REPLIES = load_csv_data()
 
@app.post("/chat")
def chat(chat_message: ChatMessage):
	reply = get_bot_reply(chat_message.message)
	return {"reply": reply}
 
def get_bot_reply(message: str) -> str:
	cleaned = message.lower().strip()
	# 1. Check the dictionary first (fastest, most reliable)
	for keyword, reply in KNOWN_REPLIES.items():
		if keyword in cleaned:
			return reply
	# 2. Check the CSV next
	if cleaned in CSV_REPLIES:
		return CSV_REPLIES[cleaned]
	# 3. Fall back to Gemini for anything we don't recognize
	return ask_gemini(message)
 
def ask_gemini(message: str) -> str:
	try:
		response = gemini_model.generate_content(message)
		return response.text
	except Exception as e:
		print("Gemini error:", e)
		return "Sorry, I'm having trouble thinking right now. Try again in a moment!"

