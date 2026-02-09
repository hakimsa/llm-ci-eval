from openai import OpenAI
import json
import os
import sys
from statistics import mean

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

THRESHOLD = 4.0

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


def judge(answer, question):
    with open("eval/judge_prompt.txt") as f:
        judge_prompt = f.read()

    full_prompt = f"""
Question: {question}
Answer: {answer}

{judge_prompt}
"""

    result = ask_llm(full_prompt)
    return json.loads(result)


with open("eval/test_cases.json") as f:
    tests = json.load(f)

scores = []

for test in tests:
    answer = ask_llm(test["question"])
    evaluation = judge(answer, test["question"])
    scores.append(evaluation["correctness"])

avg_score = mean(scores)
print(f"Average score: {avg_score}")

if avg_score < THRESHOLD:
    print("❌ Quality gate failed")
    sys.exit(1)

print("✅ Quality gate passed")
