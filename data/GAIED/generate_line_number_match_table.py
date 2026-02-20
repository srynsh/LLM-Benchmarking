import os
import json
from collections import defaultdict, Counter
from termcolor import colored

generator_data_dir = "generator_data"
validator_data_dir = "validator_data"


def get_feedback_line_number_counts(feedback_list):
    """Returns a Counter of line_number occurrences from a feedback list."""
    if not isinstance(feedback_list, list) or not feedback_list:
        return Counter()
    return Counter(str(fb["line_number"]) for fb in feedback_list if isinstance(fb, dict) and "line_number" in fb)


def load_generator_feedback():
    """Returns: dict[generator][sid] = Counter(line_number)"""
    result = defaultdict(lambda: defaultdict(Counter))
    for filename in os.listdir(generator_data_dir):
        if not filename.endswith("_feedback.json"):
            continue
        gen = filename.replace("_feedback.json", "")
        with open(os.path.join(generator_data_dir, filename), "r") as f:
            data = json.load(f)
        for entry in data:
            sid = str(entry.get("sid"))
            feedback = entry.get("feedback", [])
            result[gen][sid] = get_feedback_line_number_counts(feedback)
    return result


def load_validator_feedback():
    """Returns: dict[gen][val][sid] = Counter(line_number)"""
    import re
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for filename in os.listdir(validator_data_dir):
        m = re.match(
            r'new_labeller_gen_(.+?)_val_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$', filename)
        if not m:
            continue
        gen, val = m.groups()
        with open(os.path.join(validator_data_dir, filename), "r") as f:
            data = json.load(f)
        for entry in data:
            sid = str(entry.get("sid"))
            output = entry.get("output", {})
            if not isinstance(output, dict):
                feedback_lines = []
            else:
                feedback_lines = output.get("feedback_lines", [])
            result[gen][val][sid] = get_feedback_line_number_counts(
                feedback_lines)
    return result


def color_val(val, color, cond):
    return colored(str(val), color) if cond and val > 0 else str(val)


def main():
    gen_feedback = load_generator_feedback()
    val_feedback = load_validator_feedback()
    all_gens = sorted(set(gen_feedback.keys()) | set(val_feedback.keys()))
    all_vals = sorted(
        {val for gen in val_feedback for val in val_feedback[gen]})
    # Table header
    header = ["Generator \\ Validator"] + all_vals
    rows = []
    for gen in all_gens:
        row = [gen]
        for val in all_vals:
            matched = 0
            extra_gen = 0
            extra_val = 0
            # SIDs present in either generator or validator
            sids = set(gen_feedback[gen].keys()) | set(
                val_feedback[gen][val].keys())
            for sid in sids:
                gen_counts = gen_feedback[gen].get(sid, Counter())
                val_counts = val_feedback[gen][val].get(sid, Counter())
                all_lines = set(gen_counts.keys()) | set(val_counts.keys())
                for line in all_lines:
                    g = gen_counts.get(line, 0)
                    v = val_counts.get(line, 0)
                    matched += min(g, v)
                    extra_gen += max(g - v, 0)
                    extra_val += max(v - g, 0)
            cell = f"({matched}, {color_val(extra_gen, 'red', extra_gen > 0)}, {color_val(extra_val, 'green', extra_val > 0)})"
            row.append(cell)
        rows.append(row)
    # Print table
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=header, tablefmt="github"))
    except ImportError:
        print(",".join(header))
        for row in rows:
            print(",".join(row))


if __name__ == "__main__":
    main()
