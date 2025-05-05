import json
from tqdm import tqdm
from utils import get_claude_response, get_gemini_response, get_openai_response, parse_response, get_qwen_response, get_deepseek_response

results = []

with open('../../data/prompts/gaide_queries.json', 'r') as f:
    data = json.load(f)

# model = "gpt-4o-2024-08-06"
# model = "gpt-4o"
# model = "gpt-3.5-turbo"
model = "gpt-4o-mini-2024-07-18"
# model = "o1-mini"
# model = "o1-preview"
# model = "claude_3_opus"
# model = "claude_3.5_sonnet"
# model = "gemini-1.5-pro"
# model = "gemini-1.5-flash"
# model="qwen-plus"
# model="qwen-coder-plus"
model="deepseek-chat"

model = input("Enter model: ")

for s in tqdm(data):
    if s['sid'] <= 64:
        continue

    query = s['prompt']
    
    if model[:3] == "gpt":
        resp = get_openai_response([{"role": "user", "content": query}], model)
    elif model[:6] == "claude":
        resp = get_claude_response([{"role": "user", "content": [{"type": "text", "text": query}]}], "You are a Teaching Assistant", model)
    elif model[:6] == "gemini":
        resp = get_gemini_response(query, model)
    elif model[:4] == "qwen":
        resp = get_qwen_response([{"role": "user", "content": query}], model)
    elif model[:8] == "deepseek":
        resp = get_deepseek_response([{"role": "user", "content": query}], model)

    results.append({
        'response': resp,
        'data': parse_response(resp),
        'sid': s['sid']
    })

    # print(results)
    # break

    with open(f'./data_raw/{model}_feedback.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # break