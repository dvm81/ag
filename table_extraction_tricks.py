#!/usr/bin/env python3
"""
Tricks to improve table extraction for difficult images like IMG_4432.
"""
import base64
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import io
from openai import OpenAI
import cv2
import numpy as np

def trick1_targeted_prompting(base64_image, api_key, model="gpt-4o"):
    """
    Trick 1: Very specific prompting for tables.
    """
    client = OpenAI(api_key=api_key)
    
    prompts_to_try = [
        # Super specific
        "Look at the leftmost column of the table. What company tickers or symbols do you see listed there? List each one.",
        
        # Column by column
        "This image contains a financial table. Please read the 'Underlying' column and list all the company names or tickers you find.",
        
        # Step by step
        "1. Find the table in the image\n2. Locate the column with company information\n3. List every ticker/company name you see in that column",
        
        # With context
        "This is a gold mining sector table. Extract all mining company tickers from the 'Underlying' or leftmost data column.",
        
        # Force enumeration
        "Count and list each row in the table. For each row, tell me the company ticker or name in the first data column."
    ]
    
    results = {}
    for i, prompt in enumerate(prompts_to_try):
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }],
            temperature=0
        )
        results[f"prompt_{i+1}"] = {
            "prompt": prompt,
            "response": response.choices[0].message.content
        }
    
    return results


def trick2_image_preprocessing(image_path):
    """
    Trick 2: Various image preprocessing techniques.
    """
    # Load image
    img = Image.open(image_path)
    
    preprocessing_results = {}
    
    # 1. High contrast black & white
    img_bw = img.convert('L')
    img_bw = ImageEnhance.Contrast(img_bw).enhance(2.0)
    img_bw = ImageOps.autocontrast(img_bw, cutoff=2)
    preprocessing_results['high_contrast_bw'] = img_to_base64(img_bw)
    
    # 2. Edge detection to highlight table structure
    img_edges = img.convert('L')
    img_edges = img_edges.filter(ImageFilter.FIND_EDGES)
    img_edges = ImageEnhance.Contrast(img_edges).enhance(3.0)
    preprocessing_results['edge_detection'] = img_to_base64(img_edges)
    
    # 3. Upscale for better readability
    img_upscaled = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    preprocessing_results['upscaled_2x'] = img_to_base64(img_upscaled)
    
    # 4. Crop to just the table area (if known)
    # You'd need to adjust these coordinates for your specific image
    # img_cropped = img.crop((left, top, right, bottom))
    
    # 5. Threshold to pure black and white
    img_threshold = img.convert('L')
    img_threshold = img_threshold.point(lambda x: 0 if x < 128 else 255, '1')
    preprocessing_results['threshold_bw'] = img_to_base64(img_threshold)
    
    # 6. Sharpen text
    img_sharp = img.filter(ImageFilter.SHARPEN)
    img_sharp = img_sharp.filter(ImageFilter.SHARPEN)  # Double sharpen
    preprocessing_results['sharpened'] = img_to_base64(img_sharp)
    
    return preprocessing_results


def trick3_table_specific_preprocessing(image_path):
    """
    Trick 3: OpenCV-based table-specific preprocessing.
    """
    # Read image
    img = cv2.imread(image_path)
    
    results = {}
    
    # 1. Remove horizontal lines (keep text)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (255,255,255), 5)
    
    results['no_lines'] = img_to_base64_cv2(img)
    
    # 2. Enhance text contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l,a,b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    results['clahe_enhanced'] = img_to_base64_cv2(enhanced)
    
    return results


def trick4_multiple_api_calls(base64_image, api_key, model="gpt-4o"):
    """
    Trick 4: Multiple calls with different parameters.
    """
    client = OpenAI(api_key=api_key)
    
    strategies = [
        {"temperature": 0, "max_tokens": 500, "detail": "high"},
        {"temperature": 0.3, "max_tokens": 1000, "detail": "high"},
        {"temperature": 0, "max_tokens": 2000, "detail": "auto"},
    ]
    
    results = {}
    prompt = "Extract ALL company tickers/symbols from this financial table. Be thorough and list every single one you can see."
    
    for i, params in enumerate(strategies):
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}", 
                        "detail": params.pop("detail", "high")
                    }}
                ]
            }],
            **params
        )
        results[f"strategy_{i+1}"] = response.choices[0].message.content
    
    return results


def trick5_azure_specific_adjustments(base64_image, api_key):
    """
    Trick 5: Azure-specific optimizations.
    """
    # For Azure, you'd use:
    # from openai import AzureOpenAI
    # client = AzureOpenAI(
    #     api_key=api_key,
    #     api_version="2024-02-15-preview",  # Use latest
    #     azure_endpoint="https://your-resource.openai.azure.com"
    # )
    
    strategies = {
        # 1. Try different API versions
        "latest_api": "2024-02-15-preview",
        "stable_api": "2023-12-01-preview",
        
        # 2. Add system message for Azure
        "system_message": "You are a financial data extraction expert. Focus on accurately reading tables and extracting all ticker symbols.",
        
        # 3. Use JSON mode if available
        "response_format": {"type": "json_object"},
        
        # 4. Adjust token limits for Azure
        "max_tokens": 800,  # Azure might have different limits
    }
    
    return strategies


def img_to_base64(pil_image, format="PNG"):
    """Convert PIL image to base64."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()


def img_to_base64_cv2(cv2_image):
    """Convert OpenCV image to base64."""
    _, buffer = cv2.imencode('.png', cv2_image)
    return base64.b64encode(buffer).decode()


def run_all_tricks(image_path, api_key, model="gpt-4o"):
    """
    Run all tricks and compare results.
    """
    print("Testing various tricks to improve table extraction...")
    print("="*60)
    
    # Original image
    with open(image_path, 'rb') as f:
        original_b64 = base64.b64encode(f.read()).decode()
    
    # Trick 1: Prompting
    print("\n1. TARGETED PROMPTING:")
    prompt_results = trick1_targeted_prompting(original_b64, api_key, model)
    for key, val in prompt_results.items():
        print(f"\n{key}:")
        print(f"Prompt: {val['prompt'][:100]}...")
        print(f"Response: {val['response'][:200]}...")
    
    # Trick 2: Preprocessing
    print("\n\n2. IMAGE PREPROCESSING:")
    preprocess_results = trick2_image_preprocessing(image_path)
    for technique, img_b64 in preprocess_results.items():
        print(f"\nTrying {technique}...")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all company tickers from this table image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            temperature=0
        )
        print(f"Result: {response.choices[0].message.content[:200]}...")
    
    # Save preprocessed images for inspection
    save_preprocessed_images(preprocess_results, "preprocessed_images")
    print("\nPreprocessed images saved to 'preprocessed_images/' directory")


def save_preprocessed_images(images_dict, output_dir):
    """Save preprocessed images for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for name, img_b64 in images_dict.items():
        img_data = base64.b64decode(img_b64)
        with open(f"{output_dir}/{name}.png", 'wb') as f:
            f.write(img_data)


if __name__ == "__main__":
    # Test with IMG_4432
    if os.path.exists("IMG_4432_minimal.png"):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            run_all_tricks("IMG_4432_minimal.png", api_key)
        else:
            print("Please set OPENAI_API_KEY")
    else:
        print("IMG_4432_minimal.png not found")