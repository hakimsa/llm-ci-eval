import json
import sys
import os
import openai
from statistics import mean


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY no está definida")

openai.api_key = api_key

print("Todo OK, empezando evaluación 🚀")
