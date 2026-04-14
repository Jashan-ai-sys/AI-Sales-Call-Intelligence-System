import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_API_KEY"],
)

result = client.text_classification(
    "I like you. I love you",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
print(result)