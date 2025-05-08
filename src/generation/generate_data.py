import json
from tqdm import tqdm
from utils import get_claude_response, get_gemini_response, get_openai_response, parse_response, get_qwen_response, get_deepseek_response

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
model = "gpt-3.5-turbo"
# model = "gpt-4o-mini-2024-07-18"
# model = "o1-mini"
# model = "o1-preview"
# model = "claude_3_opus"
# model = "claude_3.5_sonnet"
# model = "claude_3.5_haiku"
# model = "gemini-1.5-pro"
# model = "gemini-1.5-flash"
# model="qwen-plus"
# model="qwen-coder-plus"
# model="deepseek-chat"

sids = [201]

####################
# Read vars
####################

if model is None:
    model = input("Enter model: ")

g_feedback = []
with open('../../data/generator/gpt-3.5-turbo_feedback.json', 'r') as f:
    g_feedback = json.load(f)

####################
# Run loop
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
    for fb in g_feedback:
        if fb['sid'] == sid:
            return fb['pid']
    return None

def get_student_code(sid):
    for fb in g_feedback:
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


for s in tqdm(data):
    sid = s['sid']
    # Skip if sid is not in sids
    if sids is not None and len(sids) != 0 and sid not in sids:
        continue

    # Get the query
    query = s['prompt']
    
    # Invoke the appropriate model
    try:
        resp_str = invoke_model(model, query)
        resp, retVal = parse_response(resp_str)
    except Exception as e:
        print(f"Error: {e}")
        continue
    
    # JSONify the response
    result = curate_result(sid, resp)
    results.append(result)

    # print(results)
    # break

####################
# Write to file
####################

    with open(f'../../data/generator/raw/{model}_feedback.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # break