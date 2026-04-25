import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def generate_quiz(text_chunks: str) -> list:
    client = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.5
    )

    prompt = f"""
You are a quiz generator. Based on the following study material, generate exactly 5 multiple choice questions.

Return ONLY a JSON array with this exact format, no extra text:
[
  {{
    "question": "Question text here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1"
  }}
]

Study Material:
{text_chunks[:3000]}
"""

    response = client.invoke(prompt)
    content = response.content.strip()

    # Clean up markdown if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    questions = json.loads(content)
    return questions