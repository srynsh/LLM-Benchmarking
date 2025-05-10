import json
import os
import numpy as np
import sys
import dataGenerator as dg 

def get_counts(feedback):
    TP, FP, ec = 0, 0, 0

    for line in feedback:
        try:
            if line['classification'] == 'valid':
                TP += 1
            elif line['classification'] == 'invalid':
                FP += 1
        except:
            ec += 1

    return TP, FP, ec

def get_counts_total(all_feedbacks):
    TP, FP, EC = 0, 0, 0

    for sub in all_feedbacks:
        try:
            labelled_feedback = sub['output']['feedback_lines']
        except:
            labelled_feedback = []

        tp, fp, ec = get_counts(labelled_feedback)
        TP += tp
        FP += fp
        EC += ec



    return TP, FP, EC

def get_counts_total_from_file(file):
    with open(file) as f:
        all_feedbacks = json.load(f)

    return get_counts_total(all_feedbacks)

def get_counts_total_from_dir(MODELS, directory):
    
    counts = {}

    vals = MODELS

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        TP, FP, EC = 0, 0, 0

        # print(file)
        gen = file.split("_gen_")[1].split("_val_")[0]

        validator = file.split("_val_")[1].split("2024")[0][:-1]

        if validator not in vals:
            validator = file.split("_val_")[1].split("2025")[0][:-1]

        tp, fp, ec = get_counts_total_from_file(f"{directory}/{file}")
        TP += tp
        FP += fp
        EC += ec

        counts[(gen, validator)] = (TP, FP, EC)


    # sort by gen, val 
    counts = sorted(counts.items(), key=lambda x: (vals.index(x[0][0]), vals.index(x[0][1])))

    return counts

def get_GV(directory="../../data/validator"): # This gives the GV array
    counts = get_counts_total_from_dir(directory)
    
    c = 0
    x = []
    i = 0
    for (gen, val), (TP, FP, EC) in counts:
        if c % 10 == 0:
            i += 1
            x.append([])
        # Avoid division by zero by checking if (TP + FP) > 0
        rate = TP / (TP + FP) if (TP + FP) > 0 else 0
        x[i - 1].append(rate)
        c += 1

    return np.array(x)

def get_VALIDATOR_COUNTS(GENS, MODELS, directory="../../data/validator"):
    gens = GENS
    vals = MODELS

    tables = np.zeros((len(gens), len(vals), 4))

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue

        gen = file.split("_gen_")[1].split("_val_")[0]
        val = file.split("_val_")[1].split("2024")[0].strip("_")
        if val not in vals:
            val = file.split("_val_")[1].split("2025")[0].strip("_")

        if gen not in gens:
            continue

        # print(f"Gen: {gen}, Val: {val}")

        iv_iv, iv_v, v_iv, v_v, ec = 0, 0, 0, 0, 0

        with open(f"{directory}/{file}", "r") as f:
            gen_labels = json.load(f)


        for i, row in enumerate(gen_labels):
            sid = row['sid']
            if gen == "gpt-4o":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_4o(sid))
            elif gen == "gpt-4-turbo":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_4_turbo(sid))
            elif gen == "gpt-3.5-turbo":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_35(sid))
            elif gen == "claude_3_opus":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_3_opus(sid))
            elif gen == "gemini-1.5-pro":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_gemini_15_pro(sid))
            elif gen == "qwen-coder-plus":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_qwen_coder_plus(sid))
            elif gen == "deepseek-chat":
                labelled_feedback, unlabelled_feedback, student_code, correct_code, question, all_testcases, prefix, suffix, _ = dg.get_data(dg.get_row_deepseek_chat(sid))
            else:
                raise Exception("Invalid gen")

            try:
                gen_lab_fb = row['output']['feedback_lines']
            except:
                gen_lab_fb = []
                
            for i in range(len(gen_lab_fb)):
                try:
                    if gen_lab_fb[i]['classification'] == 'invalid':
                        if labelled_feedback[i]['category'] in ['FP-I', 'FP-H']:
                            iv_iv += 1
                        elif labelled_feedback[i]['category'] in ['TP', 'FP-E', 'FP-R']:
                            v_iv += 1
                    elif gen_lab_fb[i]['classification'] == 'valid':
                        if labelled_feedback[i]['category'] in ['FP-I', 'FP-H']:
                            iv_v += 1
                        elif labelled_feedback[i]['category'] in ['TP', 'FP-E', 'FP-R']:
                            v_v += 1
                except:
                    ec += 1

        try:
            tables[gens.index(gen)][vals.index(val)] = [iv_iv, iv_v, v_iv, v_v]
        except:
            # print(f"Gen: {gen}, Val: {val} not found in tables")
            pass
        try:
            accuracy = (iv_iv + v_v) / (iv_iv + iv_v + v_iv + v_v)
            recall = v_v/(v_v + iv_v)
            precision = v_v/(v_v + v_iv)
            f_score = 2 * (precision * recall) / (precision + recall)
        except:
            accuracy, recall, precision, f_score = 0, 0, 0, 0

        # print(f"IV-IV: {iv_iv}, IV-V: {iv_v}, V-IV: {v_iv}, V-V: {v_v}, EC: {ec}, Total: {iv_iv + iv_v + v_iv + v_v }")
        # print(f"Precision: {precision}")

    tables = tables.astype(int)
    arr = tables.tolist()
    return arr