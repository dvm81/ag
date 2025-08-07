#!/usr/bin/env python3
"""
Extremely simple ticker extraction from base64 image.
"""
import base64
from openai import OpenAI

def extract_tickers_from_image(base64_image, api_key, model="gpt-4o"):
    """
    Barebone function to extract tickers from a base64 image.
    
    Args:
        base64_image: Base64 encoded image string
        api_key: OpenAI API key
        model: Model name (default: gpt-4o)
    
    Returns:
        String response from the model
    """
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please extract the company tickers from the image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    import os
    
    # Read an image file
    with open("extracted_images/IMG_4432.jpeg", "rb") as f:
        image_data = f.read()
    
    # Convert to base64
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Extract tickers
    result = extract_tickers_from_image(base64_image, api_key)
    print(result)