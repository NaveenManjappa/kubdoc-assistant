import requests
import os
import json

api_key = os.getenv("GROQ_API_KEY")
url ="https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(json.dumps(response.json(),indent=2))