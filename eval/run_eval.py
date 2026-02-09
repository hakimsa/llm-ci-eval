from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import sys
from statistics import mean

THRESHOLD = 0.65  # similitud mínima aceptable

model = SentenceTransformer("all-MiniLM-L6-v2")

def similarity(a, b):
    emb = model.encode([a, b])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

with open("eval/test_cases.json") as f:
    tests = json.load(f)

scores = []

for test in tests:
    # aquí simulas la respuesta del modelo candidato
    candidate_answer = test["expected_answer"]

    score = similarity(candidate_answer, test["expected_answer"])
    scores.append(score)

avg_score = mean(scores)
print(f"Average similarity score: {avg_score}")

if avg_score < THRESHOLD:
    print("❌ Quality gate failed")
    sys.exit(1)

print("✅ Quality gate passed")

