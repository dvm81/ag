#!/usr/bin/env python3
"""
Extract and display the images from our test HTML file.
"""
import base64
import os
from PIL import Image
import io
from image_utils import extract_images_from_html

def save_images_from_html():
    """Extract images from HTML and save them as viewable PNG files."""
    
    # Read the test HTML file
    with open('test_mixed_content.html', 'r') as f:
        html_content = f.read()
    
    # Extract images
    cleaned_html, images = extract_images_from_html(html_content)
    
    print(f"Found {len(images)} images in the HTML file:")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = "extracted_images"
    os.makedirs(output_dir, exist_ok=True)
    
    for img in images:
        try:
            # Decode base64 to image
            img_data = base64.b64decode(img.base64_data)
            
            # Open with PIL
            pil_img = Image.open(io.BytesIO(img_data))
            
            # Save as PNG file
            output_path = os.path.join(output_dir, f"{img.id}.png")
            pil_img.save(output_path)
            
            print(f"✅ {img.id}:")
            print(f"   Format: {img.format}")
            print(f"   Size: {pil_img.size[0]}x{pil_img.size[1]} pixels")
            print(f"   File size: {img.size_bytes:,} bytes")
            print(f"   Saved to: {output_path}")
            print(f"   Estimated tokens: {img.estimated_tokens}")
            
            # Show what this image contains based on our creation script
            if img.id == "img_001":
                print(f"   Content: Company list with tickers (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, META)")
            elif img.id == "img_002":
                print(f"   Content: Stock performance chart with AAPL, MSFT, GOOGL, TSLA")
            elif img.id == "img_003":
                print(f"   Content: Microsoft logo")
            elif img.id == "img_004":
                print(f"   Content: Apple Inc. logo")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing {img.id}: {e}")
    
    print(f"All images saved to '{output_dir}/' directory")
    print(f"You can open them with any image viewer or Preview app")
    
    # Also show extraction results mapping
    print(f"\n📊 What was extracted from each image:")
    print(f"{'='*60}")
    
    # Load the results to show what came from each image
    try:
        import json
        with open('extraction_results_with_sources.json', 'r') as f:
            results = json.load(f)
        
        # Group companies by source
        by_source = {}
        for company in results.get('companies', []):
            source_id = company.get('source_id', 'unknown')
            if source_id not in by_source:
                by_source[source_id] = []
            by_source[source_id].append(company.get('Word', 'N/A'))
        
        for source_id in sorted(by_source.keys()):
            if source_id.startswith('img_'):
                companies = by_source[source_id]
                print(f"{source_id}: {', '.join(companies)} ({len(companies)} companies)")
        
    except FileNotFoundError:
        print("No extraction results found. Run the extraction first to see what was found in each image.")

if __name__ == "__main__":
    save_images_from_html()