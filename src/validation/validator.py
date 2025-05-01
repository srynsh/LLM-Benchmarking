import sys

sys.path.append("..")
from data import get_data, get_row, get_row_4o_mini, get_row_4_1, get_row_3_opus, get_annonated_data, get_row_35, get_row_35_sonnet, get_row_gemini_15_pro, get_row_gemini_15_flash, get_row_qwen_coder_plus, get_row_deepseek_chat

import json
from utils import get_llama_response, get_gemini_response, get_claude_response, get_qwen_response, get_openai_response, get_deepseek_response, get_query, parse_response, convert_query_to_claude, convert_query_to_o1, get_cheat_query
from tqdm import tqdm
import datetime

# model = "gpt-4-turbo"
# model = "gpt-4o-2024-08-06"
# model = "gpt-3.5-turbo"
# model = "gpt-4o-mini-2024-07-18"
# model = "o1-mini"
# model = "o1-preview"
# model = "claude_3_opus"
# model = "claude_3.5_sonnet"
# model = "gemini-1.5-pro"
# model = "gemini-1.5-flash"
# model = "llama"
# model = "qwen-plus"
# model = "qwen-coder-plus"
# model = "deepseek-chat"

cheat = False

raw_responses = []

gv, giv = 0, 0

sids = [i for i in range(1, 367)]

# sids = [1, 2, 3, 4, 6, 8, 9, 12, 13, 15, 22, 23, 25, 26, 69, 98, 113, 114, 115, 116, 117, 118, 119, 120, 121, 129, 181, 207, 272, 276, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 297, 298, 301, 302, 304, 305, 307, 308, 309, 313, 315, 316, 318, 319, 321, 323, 355, 356, 357, 358, 359, 361, 362]

# gen = input("Enter generator: ")
gen = 'gpt-4o'
validator = input("Enter model: ")
# validator = 'gpt-4o-mini'

if cheat:
    ATTEMPT_NAME = f"new_labeller_gen_{gen}_val_{validator}_cheat_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
else:
    ATTEMPT_NAME = f"new_labeller_gen_{gen}_val_{validator}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"

def label_sid(sid, gen, validator):
    if gen == "gpt-4o":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row(sid))
    elif gen == "gpt-4-turbo":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_4_1(sid))
    elif gen == "gpt-3.5-turbo":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_35(sid))
    elif gen == "gpt-4o-mini":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_4o_mini(sid))
    elif gen == "claude_3_opus":
       labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_3_opus(sid))
    elif gen == "claude_3.5_sonnet":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_35_sonnet(sid))
    elif gen == "gemini-1.5-pro":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_gemini_15_pro(sid))
    elif gen == "gemini-1.5-flash":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_gemini_15_flash(sid))
    elif gen == "qwen-coder-plus":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_qwen_coder_plus(sid))
    elif gen == "deepseek-chat":
        labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, _, _, _ = get_data(get_row_deepseek_chat(sid))

    if not cheat:
        query = get_query(question, student_code, correct_code, unlabelled_feedback, all_testcases)
    else:
        ground_truth, _, _, _, _, _, _, _, _ = get_annonated_data(get_row_3_opus(sid))

        for line in ground_truth:
            if line['category'] in ['TP', 'FP-E', 'FN']:
                line['category'] = 'valid'
            elif line['category'] in ['FP-H', 'FP-I']:
                line['category'] = 'invalid'

        query = get_cheat_query(question, student_code, correct_code, unlabelled_feedback, all_testcases, ground_truth)


    if validator[:6] == "claude":
        query, sys_query = convert_query_to_claude(query)
    if validator[:2] == "o1":
        query = convert_query_to_o1(query)

    if validator == "llama":
        resp = get_llama_response(query)
    elif validator == "claude_3_opus":
        resp = get_claude_response(query, sys_query, model=validator, sleep_time=50)
    elif validator == "claude_3.5_sonnet":
        resp = get_claude_response(query, sys_query, model=validator, sleep_time=0)
    elif validator[:3] == "gpt":
        resp = get_openai_response(query, model=validator)
    elif validator[:2] == "o1":
        resp = get_openai_response(query, model=validator)
    elif validator[:6] == "gemini":
        resp = get_gemini_response(str(query), model=validator)
    elif validator[:4] == "qwen":
        resp = get_qwen_response(query, model=validator)
    elif validator[:8] == "deepseek":
        resp = get_deepseek_response(query, model=validator)

    return resp


for sid in sids:
    resp = label_sid(sid, gen, validator)

    try:
        gen_lab_fb = parse_response(resp)['feedback_lines']
        
        for i in range(len(gen_lab_fb)):
            if gen_lab_fb[i]['classification'] == 'invalid':
                giv += 1
            elif gen_lab_fb[i]['classification'] == 'valid':
                gv += 1
    except:
        pass

    raw_responses.append({
        "sid": sid,
        "raw_response": str(resp),
        "output": parse_response(resp)
    })

    with open(f"./new_logs/{ATTEMPT_NAME}.json", "w") as f:
        json.dump(raw_responses, f, indent=4)

    try:
        print(f'[**{sid}**]: precision: {gv/(gv + giv)}, TP: {gv}, FP: {giv}')
    except:
        pass