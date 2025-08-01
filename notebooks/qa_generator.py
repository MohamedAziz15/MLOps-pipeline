import openai
import json
from src.config_loader import get_env

openai.api_key = get_env("OPENAI_API_KEY")

def generate_qa_pairs(context, max_pairs=5):
    prompt = f"""
You are a helpful assistant. Read the following context and generate {max_pairs} question-answer pairs:

Context:{context}

Output as JSON list of dicts like: [{{"question": ..., "answer": ...}}]
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return json.loads(response["choices"][0]["message"]["content"])
