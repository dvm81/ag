#!/usr/bin/env python3
"""
Simple, clean function for extracting companies from a single image.
"""
import os
import json
import base64
from openai import OpenAI
from PIL import Image
import io

def extract_companies_from_image_file(image_path: str, api_key: str = None) -> list:
    """
    Extract companies from a single image file.
    
    Args:
        image_path: Path to the image file (PNG, JPG, etc.)
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
    
    Returns:
        List of company dictionaries with extracted information
        
    Example:
        companies = extract_companies_from_image_file("my_image.png")
        for company in companies:
            print(f"{company['Word']}: {company['RIC_value']}")
    """
    
    # Setup
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please provide api_key or set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Load image
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    
    # Load configuration
    with open('extraction_function_schema.json', 'r') as f:
        function_schema = json.load(f)
    
    with open('extraction_prompts.txt', 'r') as f:
        exec(f.read(), globals())
    
    # Extract companies
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system", 
                "content": "You are an expert at extracting company mentions from images. Extract all companies visible in the image with their financial identifiers and confidence levels. " + EXTRACTION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all company mentions from this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        tools=[{"type": "function", "function": function_schema}],
        tool_choice={"type": "function", "function": {"name": function_schema["name"]}},
        temperature=0
    )
    
    # Parse results
    args_string = response.choices[0].message.tool_calls[0].function.arguments
    parsed_args = json.loads(args_string)
    companies = parsed_args.get("companies", [])
    
    return companies

# Quick usage examples
if __name__ == "__main__":
    
    # Example 1: Basic usage
    print("Example 1: Extract from img_001.png")
    if os.path.exists("extracted_images/img_001.png"):
        companies = extract_companies_from_image_file("extracted_images/img_001.png")
        print(f"Found {len(companies)} companies:")
        for company in companies:
            word = company.get('Word', 'N/A')
            ric = company.get('RIC_value', 'N/A')
            confidence = company.get('RIC_confidence', 'N/A')
            print(f"  • {word}: {ric} (confidence: {confidence})")
    else:
        print("  Image not found. Run 'python view_test_images.py' first.")
    
    print("\n" + "="*50)
    
    # Example 2: Get only high-confidence companies
    print("Example 2: High-confidence companies only")
    if os.path.exists("extracted_images/img_002.png"):
        all_companies = extract_companies_from_image_file("extracted_images/img_002.png")
        high_confidence = []
        
        for company in all_companies:
            # Check if any identifier has high/very_high confidence
            identifiers = ['RIC', 'BBTicker', 'Symbol', 'ISIN', 'SEDOL']
            for id_type in identifiers:
                confidence = company.get(f'{id_type}_confidence', 'none')
                if confidence in ['high', 'very_high']:
                    high_confidence.append(company)
                    break
        
        print(f"High-confidence companies: {len(high_confidence)}/{len(all_companies)}")
        for company in high_confidence:
            print(f"  • {company.get('Word', 'N/A')}")
    else:
        print("  Image not found. Run 'python view_test_images.py' first.")
    
    print("\n" + "="*50)
    
    # Example 3: Custom API usage
    print("Example 3: Usage in your code")
    print("""
    # In your application:
    from simple_image_extractor import extract_companies_from_image_file
    
    # Extract companies
    companies = extract_companies_from_image_file("path/to/your/image.png")
    
    # Process results
    for company in companies:
        name = company['Word']
        ric = company.get('RIC_value', '')
        symbol = company.get('Symbol_value', '')
        confidence = company.get('RIC_confidence', 'none')
        
        if confidence in ['high', 'very_high']:
            print(f"Found: {name} (RIC: {ric}, Symbol: {symbol})")
    """)