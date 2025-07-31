#!/usr/bin/env python3
"""
Quick test: Extract companies from img_001.png specifically.
"""
import os
from extract_from_single_image import extract_companies_from_image, print_results

def test_img_001():
    """Test extraction specifically on img_001.png."""
    
    image_path = "extracted_images/img_001.png"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Run 'python view_test_images.py' first to extract images.")
        return
    
    print(f"🎯 Extracting companies from: {image_path}")
    print("This image contains a company list with tickers (AAPL, MSFT, TSLA, NVDA, AMZN, GOOGL, META)")
    print("-" * 80)
    
    # Extract companies using synchronous call
    result = extract_companies_from_image(image_path)
    
    # Display results
    print_results(result)
    
    # Show what we expected vs what we got
    if result.get("companies"):
        companies_found = [c.get("Word", "N/A") for c in result["companies"]]
        print(f"\n📋 Summary:")
        print(f"   Companies found: {', '.join(companies_found)}")
        print(f"   Total count: {len(companies_found)}")
    
    return result

if __name__ == "__main__":
    result = test_img_001()