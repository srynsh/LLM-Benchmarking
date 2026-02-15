import pandas as pd
import json
import ast
import numpy as np
import os
import sklearn.metrics as skm
from src.config import pathDataGAIED, pathGenerator

gpt_4o_PATH = f'{pathDataGAIED}/dataset_4o.xlsx'
gpt_4N_PATH = f'{pathDataGAIED}/dataset.xlsx'

gpt4o_benchmark = pd.read_excel(gpt_4o_PATH, sheet_name="gpt4o_refined")
GPT4_N_benchmark = pd.read_excel(gpt_4N_PATH, sheet_name="gpt4_N_refined")

questions = pd.read_excel(gpt_4o_PATH, sheet_name="problems")
correct_submissions = pd.read_excel(gpt_4o_PATH, sheet_name="correct_submissions")
buggy_submissions = pd.read_excel(gpt_4o_PATH, sheet_name="buggy_submissions")
testcases = pd.read_excel(gpt_4o_PATH, sheet_name="testcases")

feedbacks_gpt4o = gpt4o_benchmark['feedback']
feedbacks_GPT4_N = GPT4_N_benchmark['feedback']

def get_val(x):
    if (np.isnan(x)):
        return int(0)
    return int(x)

def custom_json_loads_4N(s):
    lines = s.split("\n")
    lines = lines[2:-1]

    d = {"line_number": "", "feedback": "", "category": ""}

    for line in lines:
        if (len(line) <= 8):
            continue
        line = line.split(":")
        try:
            key = line[0].strip()
            value = line[1].strip()
            value = value.split(",")[0]

            if (key == '"line_number"'):
                d["line_number"] = value[1:-1]
            elif (key == '"feedback"'):
                d["feedback"] = value[1:-1]
            elif (key == '"category"'):
                d["category"] = value[1:-1]
        except:
            print('XXXXXXXXX-----ERROR-----XXXXXXXXX')
            print(line)

    return d

def parse_feedback_4N(feedbacks):
    fba = []
    for (i, feedback) in enumerate(feedbacks):
        try:
            fba.append(json.loads(feedback))
            # sheets[i]['feedback_new'] = json.dumps(json.loads(feedback))
        except:
            jsa = []
            feedback = str(feedback)
            try:
                feedback = feedback[1:-1]
            except:
                continue
            feedback = feedback.split("},")

            for i, f in enumerate(feedback[:-1]):
                f = f + "}"
                feedback[i] = f

            for f in feedback:
                try:
                    jsa.append(json.loads(f))
                except:
                    jsa.append(custom_json_loads(f))

            fba.append(jsa)
            # sheets[i]['feedback_new'] = json.dumps(jsa)
            

    # print(len(fba))
    # st.write(fba)
    return fba

def custom_json_loads(s):
    try:
        d = ast.literal_eval(s)
        try:
            d["category"]
        except:
            d["category"] = ""
        return d
    except:
        lines = s.split("\n")
        lines = lines[2:-1]

        d = {"line_number": "", "feedback": "", "category": ""}

        for line in lines:
            if (len(line) <= 8):
                continue
            line = line.split(":")
            try:
                key = line[0].strip()
                value = line[1].strip()
                value = value.split(",")[0]

                if (key == '"line_number"'):
                    d["line_number"] = value[1:-1]
                elif (key == '"feedback"'):
                    d["feedback"] = value[1:-1]
                elif (key == '"category"'):
                    d["category"] = value[1:-1]
            except:
                print('XXXXXXXXX-----ERROR-----XXXXXXXXX')
                print(line)

        return d

def parse_feedback(feedbacks):
    fba = []
    for (i, feedback) in enumerate(feedbacks):
        try:
            fba.append(json.loads(feedback))

            for f in fba[-1]:
                try:
                    f['category']
                except:
                    f['category'] = ""
            # sheets[i]['feedback_new'] = json.dumps(json.loads(feedback))
        except:
            jsa = []
            feedback = str(feedback)
            try:
                feedback = feedback[1:-1]
            except:
                continue
            feedback = feedback.split("},")

            for i, f in enumerate(feedback[:-1]):
                f = f + "}"
                feedback[i] = f

            for f in feedback:
                try:
                    jsa.append(json.loads(f))
                except:
                    jsa.append(custom_json_loads(f))

            fba.append(jsa)

            for f in fba[-1]:
                try:
                    f['category']
                except:
                    f['category'] = ""

            # sheets[i]['feedback_new'] = json.dumps(jsa)
    return fba

def get_question(pid):
    for _, question in questions.iterrows():
        if (question['pid'] == pid):
            return question
        
def get_buggy_submissions(sid):
    for _, submission in buggy_submissions.iterrows():
        if (submission['sid'] == sid):
            return submission
        
    return None

def get_correct_submissions(sid):
    for _, submission in correct_submissions.iterrows():
        if (submission['sid'] == sid):
            return submission
        
    return None

def get_testcase(tid):
    for _, testcase in testcases.iterrows():
        if (testcase['tid'] == tid):
            return testcase

gpt4o_fba = parse_feedback(feedbacks_gpt4o)
gpt4o_benchmark['feedback'] = gpt4o_fba

gpt4N_fba = parse_feedback_4N(feedbacks_GPT4_N)
GPT4_N_benchmark['feedback'] = gpt4N_fba

def get_testcase_json(testcases, status):
    res = []
    testcases = testcases.replace("\'", "\"")
    testcases = ast.literal_eval(testcases)

    for tid in testcases:
        tc = get_testcase(int(tid))
        res.append({'status': status, 'input': tc['input'], 'expected_output': tc['output']})

    return res

def get_data(row):
    question = get_question(row['pid'])
    prefix = question['prefix']
    suffix = question['suffix']
    question = question['description']
    feedback = row['feedback']
    sid = row['sid']
    buggy_submission = get_buggy_submissions(sid)

    student_code = row['student_code']
    correct_code = row['repaired_code']
    # correct_code = get_correct_code(sid)

    unlabelled_feedback = []
    orignial_feedback = []

    for f in feedback:
        if 'category' not in f or f['category'] not in ["FN"]: # Only include feedback that is not categorized as "FN" (False Negative)
            unlabelled_feedback.append({"line_number": f['line_number'], "feedback": f['feedback']})
            orignial_feedback.append(f)
       
    failing_testcases = get_testcase_json(buggy_submission['failing_testcases'], 'FAIL')
    passing_testcases = get_testcase_json(buggy_submission['passing_testcases'], 'PASS')

    all_testcases = failing_testcases + passing_testcases

    return orignial_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, row['pid']

def get_annonated_data(row):
    question = get_question(row['pid'])
    prefix = question['prefix']
    suffix = question['suffix']
    question = question['description']
    feedback = row['feedback']
    sid = row['sid']
    buggy_submission = get_buggy_submissions(sid)

    student_code = row['student_code']
    correct_code = row['repaired_code']
    # correct_code = get_correct_code(sid)

    unlabelled_feedback = []
    orignial_feedback = []

    for f in feedback:
        if f['category'] in ["TP", "FP-H", "FP-I", "FP-E", "FN"]:
            unlabelled_feedback.append({"line_number": f['line_number'], "feedback": f['feedback']})
            orignial_feedback.append(f)

    failing_testcases = get_testcase_json(buggy_submission['failing_testcases'], 'FAIL')
    passing_testcases = get_testcase_json(buggy_submission['passing_testcases'], 'PASS')

    all_testcases = failing_testcases + passing_testcases

    return orignial_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, row['pid']

def get_correct_code(sid):
    for _, row in GPT4_N_benchmark.iterrows():
        if (row['sid'] == sid):
            return row['repaired_code']
        
def get_row(sid, modelname):
    with open(f'{pathGenerator}/{modelname}_feedback.json', 'r') as f:
        feedback = json.load(f)

    for row in feedback:
        if (row['sid'] == sid):
            return row

    raise ValueError(f"SID {sid} not found for model {modelname}")