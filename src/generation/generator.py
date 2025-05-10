import json
from tqdm import tqdm
from common.utils import get_claude_response, get_gemini_response, get_openai_response, parse_response, get_qwen_response, get_deepseek_response
import os
import traceback

####################
# Init vars
####################

results = []
sids = []
model = None

with open('../../data/prompts/gaide_queries.json', 'r') as f:
    data = json.load(f)

####################
# Config
####################

# model = "gpt-4o-2024-08-06"
# model = "gpt-4o"
# model = "gpt-3.5-turbo"
# model = "gpt-4o-mini-2024-07-18"
# model = 'gpt-4.1-mini-2025-04-14'
# model = 'gpt-4.1-2025-04-14'
# model = "o1-mini"
# model = "o1-preview"
# model = "claude_3_opus"
# model = "claude_3.5_sonnet"
# model = "claude_3.5_haiku"
# model = "gemini-1.5-pro"
# model = "gemini-1.5-flash"
# model = 'gemini-2.5-flash-preview-04-17'
model = 'gemini-2.5-pro-preview-05-06'
# model="qwen-plus"
# model="qwen-coder-plus"
# model="deepseek-chat"

####################
# Setup vars
####################
path_existing = f'../../data/generator/{model}_feedback.json'
path_gold = f'../../data/generator/gpt-3.5-turbo_feedback.json'

####################
# Read vars
####################

if model is None:
    model = input("Enter model: ")

gold_feedback = []
with open(path_gold, 'r') as f:
    gold_feedback = json.load(f)

sids = [] # By default, run on no sids
if os.path.exists(path_existing): 
    with open(path_existing, 'r') as f:
        existing_feedback = json.load(f)
        # Filter out feedbacks that are empty or None
        sids = [fb['sid'] for fb in existing_feedback if "feedback" not in fb or len(fb['feedback']) == 0]
else:
    sids = None # Run on all sids if file doesn't exist

print(f"Running on sids: {sids}")

####################
# Helpers
####################
def invoke_model(model, query):
    if model[:3] == "gpt":
        return get_openai_response([{"role": "user", "content": query}], model)
    elif model[:6] == "claude":
        return get_claude_response([{"role": "user", "content": [{"type": "text", "text": query}]}], "You are a Teaching Assistant", model)
    elif model[:6] == "gemini":
        return get_gemini_response(query, model)
    elif model[:4] == "qwen":
        return get_qwen_response([{"role": "user", "content": query}], model)
    elif model[:8] == "deepseek":
        return get_deepseek_response([{"role": "user", "content": query}], model)

def get_pid(sid):
    for fb in gold_feedback:
        if fb['sid'] == sid:
            return fb['pid']
    return None

def get_student_code(sid):
    for fb in gold_feedback:
        if fb['sid'] == sid:
            return fb['student_code']
    return None

def curate_result(sid, fb):
    feedback = []
    repaired_code = ""

    # Get the pid from g_feedback
    pid = get_pid(sid)
    student_code = get_student_code(sid)
    
    try:
        feedback = [f for f in fb['feedbacks']]
        for f in feedback:
            f['category'] = ""

        repaired_code = fb['correct_code']
    except Exception as e:
        print(f"Error: {e}")

    resp = {
        'sid': sid,
        'pid': pid,
        'feedback': feedback,
        'repaired_code': repaired_code,
        'student_code': student_code
    }    

    return resp

####################
# Loop through data
####################

for s in tqdm(data):
    sid = s['sid']
    # Skip if sid is not in sids
    if sids is not None and sid not in sids:
        continue

    # Get the query
    query = s['prompt']
    
    # Invoke the appropriate model
    try:
        resp_str = invoke_model(model, query)
        # print(f"{sid}. {resp_str}")
        resp, retVal = parse_response(resp_str)
    except Exception as e:
        traceback.print_exc()

        print(f"{sid} Response:", resp_str)
        continue
    
    # JSONify the response
    result = curate_result(sid, resp)
    results.append(result)

    # Save the results to a file
    with open(f'../../data/generator/raw/{model}_feedback.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # break