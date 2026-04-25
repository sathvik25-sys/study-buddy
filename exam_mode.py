import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def generate_exam_qa(text: str) -> list:
    client = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    prompt = f"""
You are an exam preparation assistant. Based on the following study material, generate exactly 10 important exam questions with detailed answers.

Return ONLY a JSON array with this exact format, no extra text:
[
  {{
    "question": "Important exam question here?",
    "answer": "Detailed answer here."
  }}
]

Study Material:
{text[:4000]}
"""

    response = client.invoke(prompt)
    content = response.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    return json.loads(content)