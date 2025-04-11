import json

files = [
    #  'claude_3.5_sonnet_feedback.json',
    #  'gemini-1.5-flash_feedback_fixed.json',
    #  'gemini-1.5-pro_feedback.json',
    #  'qwen-coder-plus_feedback.json'
    # 'gpt-4o-mini-2024-07-18_feedback.json'
    # 'deepseek-chat_feedback.json'
    # 'gpt-3.5-turbo_feedback_regen_missing.json'
    'claude_3_opus_feedback_regen_missing.json'
]

sids = [8, 76, 107, 108, 109, 110, 111, 163, 196, 197, 200, 201, 207, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 252, 267, 268, 269, 291, 299, 338, 339, 340, 341]
sids = [64, 86, 94, 118, 145, 163, 170, 174, 192, 222, 262]

with open('./data/gpt_4N_feedback.json', 'r') as f:
        g_feedback = json.load(f)

_feedback = []

for fb in g_feedback:
    if fb['sid'] in sids:
        _feedback.append(fb)

g_feedback = _feedback

for file in files:
    results = []

    with open(f'./data_raw/{file}', 'r') as f:
        feedback = json.load(f)

    for fb, f1 in zip(feedback, g_feedback):
        resp = {}
        try:
            resp['feedback'] = fb['data']['feedbacks']
            resp['repaired_code'] = fb['data']['correct_code']
        except:
            resp['feedback'] = []
            resp['repaired_code'] = ""

        resp['sid'] = fb['sid']
        resp['pid'] = f1['pid']
        resp['student_code'] = f1['student_code']

        results.append(resp)

    with open(f'./data/{file}', 'w') as f:
        json.dump(results, f, indent=4)
        print(f'File ./data/{file} created')
        

