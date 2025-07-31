#!/usr/bin/env python3
"""
Simple synchronous function to extract companies from a single image file.
"""
import os
import json
import base64
from openai import OpenAI
from PIL import Image
import io

def extract_companies_from_image(image_path: str, api_key: str = None, model: str = "gpt-4.1") -> dict:
    """
    Extract companies from a single image file synchronously.
    
    Args:
        image_path: Path to the image file (e.g., "extracted_images/img_001.png")
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
        model: Vision model to use (default: gpt-4.1)
    
    Returns:
        Dictionary with extracted companies and metadata
    """
    
    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please provide api_key or set OPENAI_API_KEY environment variable")
    
    # Initialize OpenAI client (synchronous)
    client = OpenAI(api_key=api_key)
    
    # Load and prepare image
    try:
        # Open and convert image to base64
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
        print(f"✅ Loaded image: {image_path}")
        print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
        
    except Exception as e:
        return {"error": f"Failed to load image: {e}", "companies": []}
    
    # Load extraction schema and prompts
    try:
        with open('extraction_function_schema.json', 'r') as f:
            function_schema = json.load(f)
        
        with open('extraction_prompts.txt', 'r') as f:
            prompts_content = f.read()
            # Execute to get the prompt variables
            exec(prompts_content, globals())
            
    except Exception as e:
        return {"error": f"Failed to load configuration: {e}", "companies": []}
    
    # Create vision-specific system prompt
    vision_system_prompt = """
You are an expert entity extraction assistant specialized in analyzing images for company mentions. Your task is to extract company mentions from images, including:

1. Company logos (even without text)
2. Company names in any text within the image
3. Stock tickers or symbols displayed
4. Companies mentioned in charts, graphs, or tables
5. Product brands that clearly indicate specific companies
6. Company names in headers, footers, or watermarks

Follow the same extraction rules as for text:
• Extract Company Mentions: Identify and extract all company names visible in the image. Prioritize recall over precision.
• Identifier: For each company mention, return exactly one identifier with the highest confidence (RIC, BBTicker, Symbol, ISIN, or SEDOL).
• IssueName: For the identified company, also return the IssueName.
• Exclusions: Ignore UBS Group and Bloomberg. Political figures are out of scope.

For images, also consider:
• Logos may represent companies even without accompanying text
• Charts/graphs may have company names on axes or legends
• Screenshots may contain company names in various UI elements
• Financial data tables often use tickers instead of full names

Always provide confidence levels and reasons that explain what you found in the image.

""" + EXTRACTION_SYSTEM_PROMPT
    
    # Make the API call
    try:
        print(f"🔍 Analyzing image with {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": vision_system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": EXTRACTION_HUMAN_PROMPT.format(
                                article_text=f"[This is an image file: {os.path.basename(image_path)}. Please analyze the visual content for company mentions.]"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            tools=[{"type": "function", "function": function_schema}],
            tool_choice={"type": "function", "function": {"name": function_schema["name"]}},
            temperature=0
        )
        
        # Parse the response
        args_string = response.choices[0].message.tool_calls[0].function.arguments
        parsed_args = json.loads(args_string)
        companies = parsed_args.get("companies", [])
        
        # Add source information
        for company in companies:
            company["source_type"] = "image"
            company["source_id"] = os.path.basename(image_path)
            company["source_context"] = f"Extracted from {os.path.basename(image_path)}"
        
        result = {
            "companies": companies,
            "metadata": {
                "image_file": os.path.basename(image_path),
                "image_path": image_path,
                "image_size": f"{img.size[0]}x{img.size[1]}",
                "model_used": model,
                "companies_found": len(companies)
            }
        }
        
        print(f"✅ Found {len(companies)} companies in the image")
        return result
        
    except Exception as e:
        return {"error": f"API call failed: {e}", "companies": []}

def print_results(result: dict):
    """Pretty print the extraction results."""
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    companies = result["companies"]
    metadata = result["metadata"]
    
    print(f"\n📊 EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"Image: {metadata['image_file']}")
    print(f"Size: {metadata['image_size']}")
    print(f"Model: {metadata['model_used']}")
    print(f"Companies found: {metadata['companies_found']}")
    print(f"{'='*60}")
    
    if not companies:
        print("No companies found in this image.")
        return
    
    for i, company in enumerate(companies, 1):
        print(f"\n{i}. {company.get('Word', 'N/A')}")
        
        # Find the best identifier
        identifiers = ['RIC', 'BBTicker', 'Symbol', 'ISIN', 'SEDOL']
        best_identifier = None
        best_confidence = None
        
        for id_type in identifiers:
            value = company.get(f'{id_type}_value')
            confidence = company.get(f'{id_type}_confidence')
            if value and confidence != 'none':
                if not best_identifier or confidence in ['very_high', 'high']:
                    best_identifier = id_type
                    best_confidence = confidence
                    break
        
        if best_identifier:
            print(f"   {best_identifier}: {company.get(f'{best_identifier}_value')} (confidence: {company.get(f'{best_identifier}_confidence')})")
            print(f"   Reason: {company.get(f'{best_identifier}_reason', 'N/A')}")
        
        issue_name = company.get('IssueName_value')
        if issue_name:
            print(f"   IssueName: {issue_name} (confidence: {company.get('IssueName_confidence', 'N/A')})")

# Example usage and main function
def main():
    """Example usage of the single image extraction."""
    
    # Check if extracted images exist
    image_dir = "extracted_images"
    if not os.path.exists(image_dir):
        print(f"❌ Directory '{image_dir}' not found.")
        print("Run 'python view_test_images.py' first to extract images.")
        return
    
    # Get available images
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    if not image_files:
        print(f"❌ No PNG images found in '{image_dir}'.")
        return
    
    print(f"📂 Available images in {image_dir}:")
    for i, img_file in enumerate(image_files, 1):
        print(f"   {i}. {img_file}")
    
    # Test with first image
    test_image = os.path.join(image_dir, image_files[0])
    print(f"\n🔍 Testing with: {test_image}")
    
    # Extract companies
    result = extract_companies_from_image(test_image)
    
    # Print results
    print_results(result)
    
    # Save results
    output_file = f"single_image_results_{image_files[0].replace('.png', '.json')}"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()