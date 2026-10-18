import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("ASU_LLM_KEY")
if not api_key:
    raise RuntimeError("ASU_LLM_KEY is missing from .env")

client = OpenAI(
    api_key=api_key,
    base_url="https://openai.rc.asu.edu/v1",
    timeout=60,
    max_retries=0,
)

models = client.models.list()

for model in sorted(models.data, key=lambda item: item.id):
    print(model.id)