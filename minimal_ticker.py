#!/usr/bin/env python3
"""
Absolute minimal ticker extraction.
"""
import os
import base64
from openai import OpenAI

# Read image
with open("extracted_images/IMG_4432.jpeg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Call OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Please extract the company tickers from the image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    temperature=0
)

print(response.choices[0].message.content)