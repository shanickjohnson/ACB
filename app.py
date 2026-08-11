from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
app = FastAPI()
 
# Allow your HTML page to talk to this server (more on this in Step 3)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)
 
# This defines the shape of the data we expect to receive
class ChatMessage(BaseModel):
	message: str
 
@app.post("/chat")
def chat(chat_message: ChatMessage):
	reply = get_bot_reply(chat_message.message)
	return {"reply": reply}
 
def get_bot_reply(message: str) -> str:
	message = message.lower()
	if "loan" in message:
		return "ACB offers Home Loans, Auto Loans and Personal Loans with competitive fixed rates from 6.5%. Would you like a pre-approval estimate?"
	elif "card" in message:
		return "We offer Classic, Gold and Business credit cards with zero annual fee for the first year."
	elif "location" in message or "branch" in message:
		return "Our branches are located in St. John's, All Saints Road, and ACB Grenada Bank. Hours: Mon–Fri, 8AM–3PM."
	else:
		return "Hello! I can help with loans, accounts, cards, branch locations, or contact info. What would you like to know?"

