from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()  # loads .env into environment variables

uri = os.getenv("MONGODB_URI")
if not uri:
    raise RuntimeError("MONGODB_URI not found. Check your .env file.")

client = MongoClient(uri, serverSelectionTimeoutMS=8000)
client.admin.command("ping")

print("✅ MongoDB Atlas connection OK")