import json
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyCBapJptXHj_CLxR5-_SazVsmfDKtYMWfQ"
genai.configure(api_key=GEMINI_API_KEY)

# print("GEMINI_API_KEY", GEMINI_API_KEY)

def get_gemini_response(prompt, model):
    try:
        client = genai.GenerativeModel(model)
        response_gemini = client.generate_content(prompt)
        return response_gemini
    except Exception as e:
        return e
    

QUERIES_PATH = "../../data/GAIED/gaide_queries.json"

with open(QUERIES_PATH) as f:
    queries = json.load(f)

for query in queries:
    resp  = get_gemini_response(query["prompt"], "gemini-1.5-flash")

    print(resp)

