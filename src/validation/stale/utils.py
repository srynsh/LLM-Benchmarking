import pandas as pd
import json
import ast
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import sklearn.metrics as skm
import time
import requests
import re
import boto3
import google.generativeai as genai
import typing_extensions as typing

load_dotenv()

CODAVERI_API_KEY = os.getenv("X_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

class feedback(typing.TypedDict):
    line_number: int
    feedback: str

class output_format(typing.TypedDict):
    correct_code: str
    feedbacks: list[feedback]

# ==============================================================================
# DEPRECATED FUNCTIONS - Use unified model interface from model.py instead
# ==============================================================================

# Legacy individual model functions - replaced by unified invoke_model()
def get_llama_response(prompt):
    try:
        llama_payload = json.dumps({
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature":0.5,
            "top_p":0.9
        })

        kwargs = {
            "modelId": "meta.llama3-1-405b-instruct-v1:0",
            "contentType": "application/json",
            "accept": "application/json",
            "body": llama_payload
        }
        
        response = bedrock_runtime.invoke_model(**kwargs)
        body = json.loads(response.get('body').read().decode('utf-8'))
        content = body['generation']
        # print(content)
        return content
    except Exception as e:
        # print(e)
        raise e

# TODO: fix the modelIds and move this to LLM
def get_claude_response(prompt, system_prompt, model, sleep_time=10):
    # time.sleep(sleep_time)
    if model == "claude_3_opus":
        modelId = "anthropic.claude-3-opus-20240229-v1:0"
        anthropic_version = "bedrock-2023-05-31"
    elif model == "claude_3.5_sonnet":
        modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        anthropic_version = "bedrock-2023-05-31"
    elif model == "claude_3.5_haiku":
        modelId = "anthropic.claude-3-5-haiku-20241022-v1:0"
        anthropic_version = "bedrock-2023-05-31"
    elif model == "claude_3.7_sonnet":
        modelId = "anthropic.claude-3-7-sonnet-20250219-v1:0"
        anthropic_version = "bedrock-2023-05-31"

    try:    
        kwargs = {
            "modelId": modelId,
            "contentType": "application/json",
            "accept": "application/json",
            "body": json.dumps({
                "anthropic_version": anthropic_version,
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": prompt
            })
        }
        
        response = bedrock_runtime.invoke_model(**kwargs)
        body = json.loads(response['body'].read())
        content = body['content'][0]['text']
        return content
    except Exception as e:
        raise e

# TODO: fix the modelIds and move this to LLM
def get_gemini_response(prompt, model):
    try:
        # List all available Gemini models
        
        client = genai.GenerativeModel(model)
        response_gemini = client.generate_content(
            prompt
        )
        # print(response_gemini)
        gemini_labels = response_gemini.text
        return gemini_labels
    except Exception as e:
        raise e

# TODO: fix the modelIds and move this to LLM
def get_openai_response(messages, model):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        response = response.choices[0].message.content
        
        return response
    except Exception as e:
        # print(e)
        raise e

def get_qwen_response(messages, model):
    '''Docs: https://help.aliyun.com/zh/model-studio/qwen-coder'''
    try:
        client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        response = response.choices[0].message.content

        return response
    except Exception as e:
        # print(e)
        raise e
    
def get_deepseek_response(messages, model):
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        response = response.choices[0].message.content

        return response
    except Exception as e:
        # print(e)
        raise e

 
def convert_query_to_claude(query):
    for q in query:
        q["content"] = [{
            "type": "text",
            "text": q["content"]
        }]

    return query[1:], query[0]["content"][0]["text"]

def convert_query_to_o1(query):
    for q in query:
        if q["role"] == "system":
            q["role"] = "user"

    return query

def parse_response(response):
    """
    DEPRECATED: Use parse_json_response() from model.py instead.
    Legacy JSON parsing function kept for backward compatibility.
    """
    try:
        op = re.search(r'```json(.*?)```', response, re.DOTALL).group(1)
        op = json.loads(op)
        return op
    except:
        return {}
    

def get_query(question, student_code, correct_code, feedback, all_testcases):
    """
    DEPRECATED: Use get_validation_prompt() from prompt.py instead.
    Legacy prompt generation function kept for backward compatibility.
    """
    query = [
        {
            "role": "system",
            "content": """**Task**: Your goal is to assess the validity of feedback provided by a Teaching Assistant (TA) on a student's incorrect Python program.
First, you will analyze the student's code to identify specific mistakes made.
Next, compare the TA's fixed version of the code and pinpoint the changes needed to correct the student's mistakes.
Finally, analyze the feedback provided by the TA and determine if it accurately addresses the student's errors.
For each feedback line, classify it as either "valid" or "invalid" and explain your reasoning.
**Output Format**: Provide your analysis in the following JSON format

```json
    {
        "mistakes": [], // List of mistakes found in the student's code 
        "fixes": [], // List of corrections proposed in the TA's fixed code
        "feedback_lines": [ 
            {
                "line_number": <integer>, // Line number referenced by the TA feedback
                "feedback": <string>, // The feedback provided by the TA
                "analysis": <string>, // Your analysis of the feedback's accuracy 
                "classification": "valid" | "invalid" // feedback validity classification
            }
        ]
    }
```
"""
        },
        {
            "role": "user",
            "content": """**Problem description**:
```text 
You have two variables, x and y, each containing a random integer value.
Your task is to write a piece of code that will exchange the values of these two variables.
This means that the value initially held by x should be in y and vice versa.
```

**Buggy student program**:
```python 
x="newy"
y="newx"
x,y=x,y
print(x)
print(y)
```
**Fixed code generated by the teaching assistant**:
```python 
x, y = y, x
print(x)
print(y)
```
**Feedback lines by the teaching assistant that could be invalid**: 
```json 
[
    {
        "line_number": 1, 
        "feedback": "You don't need to assign new strings to x and y before swapping their values. You can directly swap the values using the syntax \"x, y = y, x\"."
    }, 
    { 
        "line_number": 2,
        "feedback": "The input is a string, but you are treating it as an integer. You need to convert the string to an integer before performing mathematical operations on it."
    }
]```

**Test case results**:
```json 
[ 
    {"expression": "x == newx", "success": false},
    {"expression": "y == newy", "success": false}, 
]
```

"""
        },
        {
            "role": "assistant",
            "content": """```json
{
    "mistakes": ["The student has made a mistake when trying to swap the values of x and y"], 
    "fixes": ["Directly swap the values instead, using the syntax \"x, y = y, x\"."],
    "feedback_lines": [ 
        { 
            "line_number": 1, 
            "feedback": "You don't need to assign new strings to x and y before swapping their values. You can directly swap the values using the syntax "x, y = y, x".", 
            "analysis": "The TA has correctly identified the mistake made by student while swapping the values and its fix",
            "classification": "valid" 
        }, 
        { 
            "line_number": 2,
            "feedback": "The print statements are correct. They will print the swapped values of x and y.", 
            "analysis": "This feedback is **invalid** because it incorrectly identifies the issue (as related to input string).", 
            "classification": "invalid" 
        }
    ]
}
```
"""
        },
        {
            "role": "user",
            "content": f"""**Problem description**: ```text {question} ```
**Buggy student program**: ```python {student_code} ```
**Test case results**: ```json {all_testcases} ```
**Fixed code generated by the teaching assistant**: ```python {correct_code} ```
**Feedback by the teaching assistant that could be invalid**: ```json {feedback} ```
"""
        }
    ]

    return query

def get_cheat_query(question, student_code, correct_code, feedback, all_testcases, ground_truth):
    """
    DEPRECATED: Use get_validation_prompt_with_ground_truth() from prompt.py instead.
    Legacy prompt generation function kept for backward compatibility.
    """
    query = [
        {
            "role": "system",
            "content": """**Task**: Your goal is to assess the validity of feedback provided by a Teaching Assistant (TA) on a student's incorrect Python program.
First, you will analyze the student's code to identify specific mistakes made.
Next, compare the TA's fixed version of the code and pinpoint the changes needed to correct the student's mistakes.
Finally, analyze the feedback provided by the TA and determine if it accurately addresses the student's errors, for this you can also use the correct feedback.
For each feedback line, classify it as either "valid" or "invalid" and explain your reasoning.
**Output Format**: Provide your analysis in the following JSON format

```json
    {
        "mistakes": [], // List of mistakes found in the student's code 
        "fixes": [], // List of corrections proposed in the TA's fixed code
        "feedback_lines": [ 
            {
                "line_number": <integer>, // Line number referenced by the TA feedback
                "feedback": <string>, // The feedback provided by the TA
                "analysis": <string>, // Your analysis of the feedback's accuracy 
                "classification": "valid" | "invalid" // feedback validity classification
            }
        ]
    }
```
"""
        },
        {
            "role": "user",
            "content": """**Problem description**:
```text 
You have two variables, x and y, each containing a random integer value.
Your task is to write a piece of code that will exchange the values of these two variables.
This means that the value initially held by x should be in y and vice versa.
```

**Buggy student program**:
```python 
x="newy"
y="newx"
x,y=x,y
print(x)
print(y)
```
**Fixed code generated by the teaching assistant**:
```python 
x, y = y, x
print(x)
print(y)
```
**Feedback lines by the teaching assistant that could be invalid**: 
```json 
[
    {
        "line_number": 1, 
        "feedback": "You don't need to assign new strings to x and y before swapping their values. You can directly swap the values using the syntax \"x, y = y, x\"."
    }, 
    { 
        "line_number": 2,
        "feedback": "The input is a string, but you are treating it as an integer. You need to convert the string to an integer before performing mathematical operations on it."
    }
]```

**Feedback lines of the correct feedback**: 
```json 
[
    {
        "line_number": 1, 
        "feedback": "You don't need to assign new strings to x and y before swapping their values. You can directly swap the values using the syntax \"x, y = y, x\"."
        "category": "valid"
    }, 
    { 
        "line_number": 2,
        "feedback": "The input is a string, but you are treating it as an integer. You need to convert the string to an integer before performing mathematical operations on it."
        "category": "invalid"
    }
]```

**Test case results**:
```json 
[ 
    {"expression": "x == newx", "success": false},
    {"expression": "y == newy", "success": false}, 
]
```

"""
        },
        {
            "role": "assistant",
            "content": """```json
{
    "mistakes": ["The student has made a mistake when trying to swap the values of x and y"], 
    "fixes": ["Directly swap the values instead, using the syntax "x, y = y, x"."],
    "feedback_lines": [ 
        { 
            "line_number": 1, 
            "feedback": "You don't need to assign new strings to x and y before swapping their values. You can directly swap the values using the syntax "x, y = y, x".", 
            "analysis": "The TA has correctly identified the mistake made by student while swapping the values and its fix",
            "classification": "valid" 
        }, 
        { 
            "line_number": 2,
            "feedback": "The print statements are correct. They will print the swapped values of x and y.", 
            "analysis": "This feedback is **invalid** because it incorrectly identifies the issue (as related to input string).", 
            "classification": "invalid" 
        }
    ]
}
```
"""
        },
        {
            "role": "user",
            "content": f"""**Problem description**: ```text {question} ```
**Buggy student program**: ```python {student_code} ```
**Test case results**: ```json {all_testcases} ```
**Fixed code generated by the teaching assistant**: ```python {correct_code} ```
**Feedback by the teaching assistant that could be invalid**: ```json {feedback} ```
**Feedback lines of the correct feedback**: ```json {ground_truth} ```
"""
        }
    ]

    return query