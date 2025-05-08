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
    try:
        op = re.search(r'```json(.*?)```', response, re.DOTALL).group(1)
        op = json.loads(op)
        return op, -1 #len(op['feedbacks'])
    except:
        return [], 0