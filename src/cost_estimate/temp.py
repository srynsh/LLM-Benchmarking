import re
import json
import sys

def extract_usage_metadata(filename):
    with open(filename, 'r') as file:
        data = file.read()

    # Find all "usage_metadata": { ... } blocks
    pattern = r'"usage_metadata"\s*:\s*\{.*?\}'
    matches = re.findall(pattern, data, flags=re.DOTALL)

    usage_metadata_list = []

    for match in matches:
        try:
            # Add braces to make it a complete JSON object
            json_text = '{' + match + '}'
            parsed_json = json.loads(json_text)
            usage_metadata_list.append(parsed_json['usage_metadata'])
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            continue

    return usage_metadata_list

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python temp.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    metadata = extract_usage_metadata(filename)
    # print(json.dumps(metadata, indent=2))

    total_inputs = 0
    total_outputs = 0
    total_tokens = 0

    for item in metadata:
        total_inputs += item.get('prompt_token_count', 0)
        total_outputs += item.get('candidates_token_count', 0)
        total_tokens += item.get('total_token_count', 0)

    print(f"Total Inputs: {total_inputs}")
    print(f"Total Outputs: {total_outputs}")
    print(f"Total Tokens: {total_tokens}")
#         # print(file)