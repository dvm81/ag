#!/usr/bin/env python3
"""
Test AzureChatOpenAI with Langchain 0.3.25
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing AzureChatOpenAI with Langchain 0.3.25")
print("=" * 60)

# Method 1: Using API Key authentication
def test_azure_chat_openai_with_api_key():
    print("\nMethod 1: AzureChatOpenAI with API Key")
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0
        )
        
        print(f"✓ Created AzureChatOpenAI: {type(llm).__name__}")
        print(f"Has 'invoke' method: {hasattr(llm, 'invoke')}")
        
        # Test invoke
        if hasattr(llm, 'invoke'):
            response = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say hello in one word")
            ])
            print(f"Response type: {type(response).__name__}")
            print(f"Response content: {response.content}")
            return True
        else:
            print("✗ No invoke method found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Method 2: Using Azure AD Token Provider
def test_azure_chat_openai_with_token():
    print("\nMethod 2: AzureChatOpenAI with Azure AD Token")
    
    try:
        # Example token provider (you would use your actual one)
        def get_azure_token():
            # This is where you'd implement your token retrieval
            # For example, using azure-identity library
            return "your-azure-ad-token"
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_ad_token_provider=get_azure_token,  # Your token provider function
            temperature=0
        )
        
        print(f"✓ Created AzureChatOpenAI with token: {type(llm).__name__}")
        print(f"Has 'invoke' method: {hasattr(llm, 'invoke')}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Method 3: With custom headers
def test_azure_chat_openai_with_headers():
    print("\nMethod 3: AzureChatOpenAI with custom headers")
    
    try:
        # Your custom headers
        default_headers = {
            "User-Agent": "MyApp/1.0",
            "Custom-Header": "custom-value"
        }
        
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", "your-api-key"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0,
            default_headers=default_headers
        )
        
        print(f"✓ Created AzureChatOpenAI with headers: {type(llm).__name__}")
        print(f"Has 'invoke' method: {hasattr(llm, 'invoke')}")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

# Check what methods are available
def inspect_azure_chat_openai():
    print("\nInspecting AzureChatOpenAI methods:")
    
    try:
        llm = AzureChatOpenAI(
            azure_endpoint="https://dummy.openai.azure.com",  # Dummy for inspection
            api_key="dummy-key",
            api_version="2023-05-15",
            azure_deployment="gpt-4",
            temperature=0
        )
        
        methods = [method for method in dir(llm) if not method.startswith('_') and callable(getattr(llm, method))]
        print(f"Available methods: {methods[:10]}...")  # Show first 10
        
        important_methods = ['invoke', 'generate', 'predict', 'predict_messages', '__call__']
        print("\nImportant methods check:")
        for method in important_methods:
            has_method = hasattr(llm, method)
            print(f"  {method}: {'✓' if has_method else '✗'}")
            
    except Exception as e:
        print(f"Error during inspection: {e}")

if __name__ == "__main__":
    # Check available credentials
    has_endpoint = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))
    has_api_key = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    has_deployment = bool(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
    
    print(f"AZURE_OPENAI_ENDPOINT: {'✓' if has_endpoint else '✗'}")
    print(f"AZURE_OPENAI_API_KEY: {'✓' if has_api_key else '✗'}")
    print(f"AZURE_OPENAI_DEPLOYMENT_NAME: {'✓' if has_deployment else '✗'}")
    
    # Run tests
    inspect_azure_chat_openai()
    
    if has_endpoint and has_api_key and has_deployment:
        test1 = test_azure_chat_openai_with_api_key()
        test2 = test_azure_chat_openai_with_headers()
    else:
        print("\nSkipping actual API tests - set environment variables to test with real API")
        test1 = test2 = False
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("AzureChatOpenAI is the recommended way to use Azure OpenAI with Langchain")
    print("It should have the 'invoke' method and work directly with your setup")
    print("\nRequired environment variables:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_API_KEY (or use azure_ad_token_provider)")
    print("- AZURE_OPENAI_DEPLOYMENT_NAME")
    print("- AZURE_OPENAI_API_VERSION (optional, defaults to 2023-05-15)")
