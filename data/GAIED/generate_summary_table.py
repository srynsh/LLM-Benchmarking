import os
import json
from collections import defaultdict
from tabulate import tabulate
from termcolor import colored


def parse_generator_data(directory):
    """Parses generator data and returns a summary table."""
    summary = {}

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            model_name = filename.replace("_feedback.json", "")
            file_path = os.path.join(directory, filename)

            with open(file_path, "r") as file:
                data = json.load(file)

            total_lines = 0
            non_fn_lines = 0
            category_counts = defaultdict(int)

            for entry in data:
                feedback_list = entry.get("feedback", [])
                if isinstance(feedback_list, list):
                    for feedback in feedback_list:
                        if 'line_number' in feedback:
                            total_lines += 1
                            if feedback.get("category") != "FN":
                                non_fn_lines += 1
                                category = feedback.get(
                                    "category", "not annotated") or "not annotated"
                                category_counts[category] += 1

            summary[model_name] = {
                "Total Lines": total_lines,
                "Non-FN Lines": non_fn_lines,
                "not annotated": category_counts.get("not annotated", 0),
                **{k: v for k, v in category_counts.items() if k != "not annotated"}
            }

    return summary


def generate_table(summary):
    """Generates a table string from the summary dictionary."""
    headers = ["Model", "Total Lines", "Non-FN Lines"] + \
        list({key for model in summary.values()
             for key in model.keys() if key not in ["Total Lines", "Non-FN Lines"]})
    rows = []

    for model, data in summary.items():
        row = [
            model,
            data.get("Total Lines", 0),
            data.get("Non-FN Lines", 0)
        ]
        for header in headers[3:]:
            value = data.get(header, 0)
            if header == "not annotated" and value != data.get("Total Lines", 0) and value != 0:
                value = colored(str(value), "red")
            row.append(value)

        rows.append(row)

    table = [headers] + rows
    return table


def save_table_as_csv(table, output_path):
    """Saves the table as a CSV file."""
    with open(output_path, "w") as file:
        for row in table:
            file.write(",".join(map(str, row)) + "\n")


def print_table(table):
    """Prints the table in a formatted way using tabulate."""
    headers = table[0]
    rows = table[1:]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def get_non_fn_lines(generator_data_dir):
    non_fn_counts = {}
    for filename in os.listdir(generator_data_dir):
        if not filename.endswith("_feedback.json"):
            continue
        model = filename.replace("_feedback.json", "")
        filepath = os.path.join(generator_data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        count = 0
        for entry in data:
            feedback_list = entry.get("feedback", [])
            for fb in feedback_list:
                if isinstance(fb, dict) and 'line_number' in fb and fb.get('category', None) != 'FN':
                    count += 1
        non_fn_counts[model] = count
    return non_fn_counts


def parse_validator_files(validator_data_dir):
    import re
    results = defaultdict(lambda: defaultdict(
        lambda: {'valid': 0, 'invalid': 0, 'other': 0, 'total': 0}))
    all_validators = set()
    for filename in os.listdir(validator_data_dir):
        match = re.match(
            r'new_labeller_gen_(.+?)_val_(.+?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json$', filename
        )
        if not match:
            continue
        gen_model, val_model = match.groups()
        all_validators.add(val_model)
        filepath = os.path.join(validator_data_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for entry in data:
            output = entry.get('output', {})
            if not isinstance(output, dict):
                continue
            feedback_lines = output.get('feedback_lines', [])
            if not isinstance(feedback_lines, list):
                continue
            for fb in feedback_lines:
                if not isinstance(fb, dict):
                    continue
                classification = fb.get('classification', '')
                if not isinstance(classification, str):
                    continue
                classification = classification.lower()
                results[gen_model][val_model]['total'] += 1
                if classification == 'valid':
                    results[gen_model][val_model]['valid'] += 1
                elif classification == 'invalid':
                    results[gen_model][val_model]['invalid'] += 1
                else:
                    results[gen_model][val_model]['other'] += 1
    return results, all_validators


def color_tuple(valid, invalid, other):
    valid_str = colored(str(valid), 'green') if valid else str(valid)
    invalid_str = colored(str(invalid), 'red') if invalid else str(invalid)
    other_str = colored(str(other), 'blue') if other else str(other)
    return f"({valid_str}, {invalid_str}, {other_str})"


def generate_validator_table(generator_data_dir, validator_data_dir):
    non_fn_counts = get_non_fn_lines(generator_data_dir)
    validator_results, all_validators = parse_validator_files(
        validator_data_dir)
    all_validators = sorted(all_validators)
    all_generators = sorted(set(non_fn_counts.keys()) |
                            set(validator_results.keys()))
    headers = ["Generator", "Non-FN Lines"] + all_validators
    rows = []
    for gen in all_generators:
        row = [gen, non_fn_counts.get(gen, 0)]
        for val in all_validators:
            res = validator_results.get(gen, {}).get(
                val, {'valid': 0, 'invalid': 0, 'other': 0, 'total': 0})
            cell = f"{res['total']} {color_tuple(res['valid'], res['invalid'], res['other'])}"
            row.append(cell)
        rows.append(row)
    return [headers] + rows


def main():
    generator_data_dir = "generator_data"
    validator_data_dir = "validator_data"
    output_dir = "summary_outputs"

    os.makedirs(output_dir, exist_ok=True)

    # Generator summary table
    summary = parse_generator_data(generator_data_dir)
    table = generate_table(summary)
    output_path = os.path.join(output_dir, "generator_summary_table.csv")
    save_table_as_csv(table, output_path)
    print(f"Summary table saved to {output_path}")
    print_table(table)

    # Validator summary table
    validator_table = generate_validator_table(
        generator_data_dir, validator_data_dir)
    validator_output_path = os.path.join(
        output_dir, "generator_validator_summary_table.csv")
    save_table_as_csv(validator_table, validator_output_path)
    print(f"\nValidator summary table saved to {validator_output_path}")
    print_table(validator_table)


if __name__ == "__main__":
    main()
