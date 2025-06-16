#!/usr/bin/env python3
"""
Multi-Agent System for MSFT Stock Prediction
Using LangGraph with prompt-only agents
"""

import os
import sys
import json
import subprocess
import tempfile
from typing import Dict, Any, List, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Configure logging
def setup_logging():
    """Configure comprehensive logging for the multi-agent system"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up file handler for detailed logs
    file_handler = logging.FileHandler(log_dir / f'msft_agents_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Set up console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Note: Environment paths - update these if running in a different environment
# Comment out these lines if not using the specific conda environment
# os.environ['PATH'] = f"/Users/dimitermilushev/miniforge3/envs/msft_agents/bin:{os.environ['PATH']}"
# sys.path.insert(0, '/Users/dimitermilushev/miniforge3/envs/msft_agents/lib/python3.10/site-packages')

# Define the state for our workflow
class GraphState(TypedDict):
    """State passed between agents"""
    messages: List[str]
    eda_output: str
    feature_output: str 
    model_output: str
    eval_output: str
    logs: Dict[str, Any]

# Working wrapper for raw AzureOpenAI client
class WorkingAzureWrapper:
    """Wrapper that makes raw AzureOpenAI client work with Langchain"""
    
    def __init__(self, azure_endpoint, api_key, api_version, deployment_name, default_headers=None, azure_ad_token_provider=None):
        from openai import AzureOpenAI
        
        # Create the raw Azure client using your exact setup
        if api_key:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                default_headers=default_headers or {}
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                azure_ad_token_provider=azure_ad_token_provider,
                default_headers=default_headers or {}
            )
        
        self.deployment_name = deployment_name
        logger.debug(f"Created AzureOpenAI client with deployment: {deployment_name}")
    
    def invoke(self, messages):
        """Convert Langchain messages and call Azure client"""
        
        # Convert Langchain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else:
                # Handle unknown message types
                openai_messages.append({"role": "user", "content": str(msg)})
        
        logger.debug(f"Converted {len(messages)} Langchain messages to OpenAI format")
        
        # Call Azure OpenAI client using the method it actually has
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=openai_messages,
            temperature=0
        )
        
        # Return as AIMessage for Langchain compatibility
        content = response.choices[0].message.content
        return AIMessage(content=content)
    
    # Add other methods for compatibility
    def generate(self, message_lists):
        """Generate method for compatibility"""
        results = []
        for messages in message_lists:
            response = self.invoke(messages)
            results.append(response)
        return results
    
    def predict(self, text, **kwargs):
        """Simple predict method"""
        response = self.invoke([HumanMessage(content=text)])
        return response.content
    
    def predict_messages(self, messages, **kwargs):
        """Predict messages method"""
        return self.invoke(messages)
    
    def __call__(self, messages, **kwargs):
        """Make the wrapper callable"""
        return self.invoke(messages)

# Initialize the LLM
def get_llm():
    """Get the LLM instance - supports both Azure OpenAI and standard OpenAI"""
    
    # Check if using Azure OpenAI
    if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_ENDPOINT"):
        logger.info("Using Azure OpenAI with WorkingAzureWrapper")
        
        # Get configuration from environment variables
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
        # Validate required parameters
        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME must be set")
        
        # Custom headers (you can customize these based on your setup)
        default_headers = {
            # Add any custom headers you need here
            # "User-Agent": "MSFT-Agents/1.0",
        }
        
        # Support for token provider (if you're using Azure AD authentication)
        azure_ad_token_provider = None  # You can set this if using token-based auth
        
        try:
            # Try AzureChatOpenAI first (the "proper" way)
            try:
                from langchain_openai import AzureChatOpenAI
                
                if api_key:
                    llm = AzureChatOpenAI(
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version,
                        azure_deployment=deployment_name,
                        temperature=0,
                        default_headers=default_headers
                    )
                else:
                    if azure_ad_token_provider is None:
                        raise ValueError("Either AZURE_OPENAI_API_KEY or azure_ad_token_provider must be set")
                    
                    llm = AzureChatOpenAI(
                        azure_endpoint=endpoint,
                        api_version=api_version,
                        azure_deployment=deployment_name,
                        temperature=0,
                        default_headers=default_headers,
                        azure_ad_token_provider=azure_ad_token_provider
                    )
                
                # Test if it actually works
                if hasattr(llm, 'invoke'):
                    logger.debug("✓ Successfully initialized AzureChatOpenAI")
                    return llm
                else:
                    raise AttributeError("AzureChatOpenAI missing invoke method")
                    
            except Exception as e:
                logger.warning(f"AzureChatOpenAI failed: {e}")
                logger.info("Falling back to WorkingAzureWrapper...")
                
                # Fallback to our working wrapper
                wrapper = WorkingAzureWrapper(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    deployment_name=deployment_name,
                    default_headers=default_headers,
                    azure_ad_token_provider=azure_ad_token_provider
                )
                
                logger.debug("✓ Successfully created WorkingAzureWrapper")
                return wrapper
                
        except Exception as e:
            logger.error(f"All Azure OpenAI initialization methods failed: {e}")
            logger.error("Please check your Azure OpenAI configuration:")
            logger.error(f"  Endpoint: {endpoint}")
            logger.error(f"  Deployment: {deployment_name}")
            logger.error(f"  API Version: {api_version}")
            logger.error(f"  Has API Key: {bool(api_key)}")
            raise
    
    # Standard OpenAI
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("No API credentials found")
            raise ValueError("Please set either OPENAI_API_KEY or AZURE_OPENAI_* environment variables")
        
        logger.info("Using standard OpenAI")
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

# Safe wrapper for LLM invocation
def safe_llm_invoke(llm, messages):
    """Safely invoke LLM with proper error handling for different Langchain versions and Azure OpenAI"""
    logger.debug(f"Invoking LLM with {len(messages)} messages")
    logger.debug(f"Message types: {[type(m).__name__ for m in messages]}")
    logger.debug(f"LLM type: {type(llm).__name__}")
    
    # Check if LLM has invoke method
    if not hasattr(llm, 'invoke'):
        logger.error(f"LLM object ({type(llm).__name__}) has no 'invoke' method")
        logger.debug(f"Available methods: {[m for m in dir(llm) if not m.startswith('_') and callable(getattr(llm, m))]}")
        
        # Try alternative methods
        if hasattr(llm, 'generate'):
            logger.info("Using 'generate' method instead of 'invoke'")
            result = llm.generate([messages])
            return result.generations[0][0].text
        elif hasattr(llm, 'predict_messages'):
            logger.info("Using 'predict_messages' method instead of 'invoke'")
            result = llm.predict_messages(messages)
            return result.content if hasattr(result, 'content') else str(result)
        elif hasattr(llm, '__call__'):
            logger.info("Using direct call method instead of 'invoke'")
            result = llm(messages)
            if isinstance(result, AIMessage):
                return result.content
            elif hasattr(result, 'content'):
                return result.content
            else:
                return str(result)
        else:
            raise AttributeError(f"LLM object has no suitable method for invocation. Available methods: {[m for m in dir(llm) if not m.startswith('_')]}")
    
    # Try invoke method
    try:
        response = llm.invoke(messages)
        
        logger.debug(f"Response type: {type(response).__name__}")
        
        # Extract content based on response type
        if isinstance(response, AIMessage):
            return response.content
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            # Try to convert to string as last resort
            return str(response)
            
    except AttributeError as e:
        error_str = str(e)
        logger.error(f"AttributeError during LLM invocation: {error_str}")
        
        # Check if it's the specific 'role' attribute error
        if "'SystemMessage' object has no attribute 'role'" in error_str or \
           "'HumanMessage' object has no attribute 'role'" in error_str or \
           "'AIMessage' object has no attribute 'role'" in error_str:
            
            logger.info("Detected 'role' attribute error - using message dict conversion")
            
            # Convert messages to dicts with proper role mapping
            message_dicts = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    message_dicts.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    message_dicts.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    message_dicts.append({"role": "assistant", "content": msg.content})
                else:
                    # Fallback for unknown message types
                    message_dicts.append({"role": "user", "content": str(msg)})
            
            logger.debug(f"Converted {len(message_dicts)} messages to dicts")
            
            # Try with message dicts
            response = llm.invoke(message_dicts)
            
            if isinstance(response, AIMessage):
                return response.content
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        else:
            # Different AttributeError - try general dict conversion anyway
            logger.warning("Different AttributeError - trying dict conversion")
            try:
                message_dicts = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        role = "user"  # default
                        if isinstance(msg, SystemMessage) or (hasattr(msg, 'type') and msg.type == 'system'):
                            role = "system"
                        elif isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                            role = "user"
                        elif isinstance(msg, AIMessage) or (hasattr(msg, 'type') and msg.type == 'ai'):
                            role = "assistant"
                        message_dicts.append({"role": role, "content": msg.content})
                    else:
                        message_dicts.append({"role": "user", "content": str(msg)})
                
                response = llm.invoke(message_dicts)
                
                if isinstance(response, AIMessage):
                    return response.content
                elif hasattr(response, 'content'):
                    return response.content
                else:
                    return str(response)
            except Exception as e2:
                logger.error(f"Alternative invocation also failed: {e2}")
                raise e  # Re-raise original error
            
    except Exception as e:
        logger.error(f"Unexpected error during LLM invocation: {type(e).__name__}: {e}")
        raise

# Helper function to execute Python code with debugging
def execute_and_debug_code(code: str, filename: str, llm, max_attempts: int = 3) -> tuple[str, str, str]:
    """Execute Python code and debug if errors occur. Returns (stdout, stderr, final_code)"""
    logger.info(f"Executing and debugging Python code: {filename}")
    
    current_code = code
    attempt = 1
    
    while attempt <= max_attempts:
        logger.info(f"Attempt {attempt}/{max_attempts} for {filename}")
        
        # Write and execute code
        with open(filename, 'w') as f:
            f.write(current_code)
        
        try:
            logger.debug(f"Running subprocess for {filename}")
            result = subprocess.run(
                [sys.executable, filename],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            stdout, stderr = result.stdout, result.stderr
            
            # If successful, return
            if not stderr or "Warning" in stderr:
                logger.info(f"{filename} executed successfully on attempt {attempt}")
                return stdout, stderr, current_code
            
            # If errors and not last attempt, try to debug
            if stderr and attempt < max_attempts:
                logger.warning(f"Errors in {filename}, attempting to debug...")
                logger.debug(f"Error: {stderr}")
                
                debug_prompt = f"""
You are debugging a Python script that encountered an error. Here is the original code:

```python
{current_code}
```

Error encountered:
{stderr}

Please provide a corrected version of the COMPLETE script that fixes this error. 
Common issues to check:
1. Date columns in correlation calculations (use numeric_cols = data.select_dtypes(include=[np.number]).columns)
2. Import issues (use 'import joblib' not 'from sklearn.externals import joblib')
3. Missing column handling when dropping Date/Target_return columns (use errors='ignore')
4. Pickle vs joblib consistency
5. For "could not convert string to float" errors with dates:
   - Make sure Date column is parsed as datetime: pd.to_datetime(data['Date'])
   - Exclude Date column from numeric operations
   - Drop Date column before scaling or modeling
6. For correlation matrix: ALWAYS use only numeric columns
7. For any DataFrame operations that might include Date: explicitly exclude it

Return ONLY the corrected Python code, nothing else.
"""
                
                try:
                    corrected_code = safe_llm_invoke(llm, [
                        SystemMessage(content=debug_prompt),
                        HumanMessage(content="Fix the code")
                    ]).strip()
                    if corrected_code.startswith("```python"):
                        corrected_code = corrected_code[9:]
                    if corrected_code.endswith("```"):
                        corrected_code = corrected_code[:-3]
                    
                    current_code = corrected_code
                    logger.info(f"Generated corrected code for {filename}")
                    
                except Exception as debug_error:
                    logger.error(f"Debug attempt failed: {debug_error}")
                    break
            else:
                break
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout expired while executing {filename}")
            return "", "Execution timeout after 300 seconds", current_code
        except Exception as e:
            logger.error(f"Error executing {filename}: {str(e)}")
            return "", str(e), current_code
        
        attempt += 1
    
    # Return final attempt result
    return stdout, stderr, current_code

# Helper function to execute Python code (legacy)
def execute_python_code(code: str, filename: str) -> tuple[str, str]:
    """Execute Python code and return stdout and stderr"""
    logger.info(f"Executing Python code: {filename}")
    logger.debug(f"Code preview (first 200 chars): {code[:200]}...")
    
    with open(filename, 'w') as f:
        f.write(code)
    
    try:
        logger.debug(f"Running subprocess for {filename}")
        result = subprocess.run(
            ["/Users/dimitermilushev/miniforge3/envs/msft_agents/bin/python", filename],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.stdout:
            logger.debug(f"STDOUT from {filename}: {result.stdout[:500]}...")
        if result.stderr:
            logger.warning(f"STDERR from {filename}: {result.stderr}")
            
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while executing {filename}")
        return "", "Execution timeout after 300 seconds"
    except Exception as e:
        logger.error(f"Error executing {filename}: {str(e)}")
        return "", str(e)

# Helper function to save script content to logs
def add_file_content(logs: dict, filename: str, log_key: str):
    try:
        with open(filename, "r") as f:
            content = f.read()
    except Exception as e:
        content = f"Error reading {filename}: {e}"
    logs[log_key] = content

# EDA Agent
def eda_agent(state: GraphState) -> GraphState:
    """EDA Agent - Performs exploratory data analysis"""
    logger.info("=" * 60)
    logger.info("Starting EDA Agent")
    
    eda_prompt = """You are a data analysis expert agent tasked with performing exploratory data analysis on Microsoft (MSFT) stock data.

Your task is to write a complete Python script called EDA.py that:
1. Loads the MSFT training data from 'data/MSFT_train.csv'
2. IMPORTANT: Parse the Date column as datetime: data['Date'] = pd.to_datetime(data['Date'])
3. Performs comprehensive exploratory data analysis including:
   - Data shape and basic info
   - Statistical summary of all columns
   - Check for missing values
   - Analyze the distribution of the target variable (Target_return)
   - Create visualizations for price trends, volume, and returns
   - IMPORTANT: For correlation analysis, use ONLY numeric columns:
     numeric_cols = data.select_dtypes(include=[np.number]).columns
     correlation_matrix = data[numeric_cols].corr()
   - Check for any data quality issues
4. Save all visualizations as PNG files with descriptive names
5. Print key insights and findings

The data contains: Date (string that needs parsing), Open, High, Low, Close, Volume, Target_return (log return for next day)

CRITICAL: The Date column is NOT numeric. Always exclude it from correlation calculations and numeric operations.

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.
The script should be production-ready and well-commented.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for EDA Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with EDA prompt")
        code = safe_llm_invoke(llm, [
            SystemMessage(content=eda_prompt),
            HumanMessage(content="Generate the EDA.py script")
        ]).strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        
        logger.info("LLM generated EDA.py script successfully")
        
        # Execute and debug the code
        stdout, stderr, final_code = execute_and_debug_code(code, "EDA.py", llm, max_attempts=3)
        
        # Check for errors
        if stderr and "Warning" not in stderr:
            logger.error(f"EDA Agent encountered errors after debugging: {stderr}")
        else:
            logger.info("EDA Agent executed successfully")
        
        # Update state
        state["eda_output"] = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        state["messages"].append(f"EDA Agent completed. Output: {stdout[:500]}...")
        
        # Log the agent details
        if "logs" not in state:
            state["logs"] = {}
        
        state["logs"]["EDA_Agent"] = {
            "prompt": eda_prompt,
            "output_log": state["eda_output"],
            "final_code": final_code
        }
        
        logger.info("EDA Agent completed")
        return state
        
    except Exception as e:
        logger.error(f"Fatal error in EDA Agent: {str(e)}")
        state["eda_output"] = f"ERROR: {str(e)}"
        state["messages"].append(f"EDA Agent failed: {str(e)}")
        return state

# Feature Engineering Agent  
def feature_engineering_agent(state: GraphState) -> GraphState:
    """Feature Engineering Agent - Creates features for the model"""
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Agent")
    
    feature_prompt = """You are a feature engineering expert agent tasked with creating features for Microsoft (MSFT) stock prediction.

Your task is to write a complete Python script called FEATURE.py that:
1. Loads the MSFT data from all three files: 'data/MSFT_train.csv', 'data/MSFT_val.csv', 'data/MSFT_test.csv'
2. IMPORTANT: Parse Date columns immediately after loading:
   - train_data['Date'] = pd.to_datetime(train_data['Date'])
   - val_data['Date'] = pd.to_datetime(val_data['Date'])
   - test_data['Date'] = pd.to_datetime(test_data['Date'])
3. Creates meaningful features for predicting next-day log returns, including:
   - Technical indicators (moving averages, RSI, MACD, Bollinger Bands, etc.)
   - Price-based features (returns, log returns, price ratios)
   - Volume-based features (volume moving averages, volume ratios)
   - Volatility features (rolling standard deviation, ATR)
   - Momentum features
   - Any other relevant financial features
4. Handles the temporal nature of the data properly (no look-ahead bias)
5. CRITICAL: Before saving, ensure Date column is included but Target_return is in the correct place
6. Saves the engineered features for each dataset as:
   - 'data/MSFT_train_features.csv'
   - 'data/MSFT_val_features.csv'  
   - 'data/MSFT_test_features.csv'
7. Prints the list of created features and their importance/relevance

The target variable is 'Target_return' (log return for next day).
The Date column should be kept for reference but excluded from feature calculations.

IMPORTANT: Handle NaN values properly - features might have NaN for initial rows due to rolling calculations.

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.
Ensure no data leakage between train/val/test sets.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Feature Engineering Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Feature Engineering prompt")
        code = safe_llm_invoke(llm, [
            SystemMessage(content=feature_prompt),
            HumanMessage(content="Generate the FEATURE.py script")
        ]).strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        
        logger.info("LLM generated FEATURE.py script successfully")
        
        # Execute and debug the code
        stdout, stderr, final_code = execute_and_debug_code(code, "FEATURE.py", llm, max_attempts=3)
        
        # Check for errors
        if stderr and "Warning" not in stderr:
            logger.error(f"Feature Engineering Agent encountered errors after debugging: {stderr}")
            # Check if it's a missing module error
            if "ModuleNotFoundError" in stderr:
                logger.error("Missing required Python module. Please install missing dependencies.")
        else:
            logger.info("Feature Engineering Agent executed successfully")
            # Check if feature files were created
            feature_files = ['data/MSFT_train_features.csv', 'data/MSFT_val_features.csv', 'data/MSFT_test_features.csv']
            for file in feature_files:
                if os.path.exists(file):
                    logger.info(f"Successfully created: {file}")
                else:
                    logger.warning(f"Expected file not created: {file}")
        
        # Update state
        state["feature_output"] = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        state["messages"].append(f"Feature Engineering Agent completed. Output: {stdout[:500]}...")
        
        # Log the agent details
        state["logs"]["FeatureEngineering_Agent"] = {
            "prompt": feature_prompt,
            "output_log": state["feature_output"],
            "final_code": final_code
        }
        
        logger.info("Feature Engineering Agent completed")
        return state
        
    except Exception as e:
        logger.error(f"Fatal error in Feature Engineering Agent: {str(e)}")
        state["feature_output"] = f"ERROR: {str(e)}"
        state["messages"].append(f"Feature Engineering Agent failed: {str(e)}")
        return state

# Modeling Agent
def modeling_agent(state: GraphState) -> GraphState:
    """Modeling Agent - Trains and validates models"""
    logger.info("=" * 60)
    logger.info("Starting Modeling Agent")
    
    model_prompt = """You are a machine learning expert agent tasked with building models to predict Microsoft (MSFT) stock returns.

Your task is to write a complete Python script called MODEL.py that:
1. Loads the feature-engineered data from:
   - 'data/MSFT_train_features.csv'
   - 'data/MSFT_val_features.csv'
2. CRITICAL Data Preparation:
   - Parse Date columns if present: pd.to_datetime(data['Date']) 
   - Drop Date column before modeling: X = data.drop(columns=['Date', 'Target_return'], errors='ignore')
   - Handle NaN values properly (dropna or fillna)
   - Scale features if needed (save the scaler!)
3. Trains multiple models to predict 'Target_return', including:
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost (use n_jobs=1 to avoid issues)
   - LightGBM
   - Any other suitable models
4. Performs hyperparameter tuning using the validation set
5. Evaluates each model on the validation set using appropriate metrics (RMSE, MAE, R2)
6. Selects the best model based on validation performance
7. IMPORTANT: Save files using joblib (not pickle):
   - import joblib
   - joblib.dump(best_model, 'best_model.pkl')
   - joblib.dump(scaler, 'scaler.pkl')
8. Prints model performance comparisons and feature importances

Important: 
- Do NOT use the test set for training or model selection. Only train/val sets.
- Ensure the Date column is excluded from features (it's not numeric!)
- Use errors='ignore' when dropping columns to handle missing columns gracefully

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Modeling Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Modeling prompt")
        code = safe_llm_invoke(llm, [
            SystemMessage(content=model_prompt),
            HumanMessage(content="Generate the MODEL.py script")
        ]).strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        
        logger.info("LLM generated MODEL.py script successfully")
        
        # Execute and debug the code
        stdout, stderr, final_code = execute_and_debug_code(code, "MODEL.py", llm, max_attempts=3)
        
        # Check for errors
        if stderr and "Warning" not in stderr:
            logger.error(f"Modeling Agent encountered errors after debugging: {stderr}")
            if "libxgboost.dylib" in stderr or "libomp" in stderr:
                logger.error("XGBoost requires libomp on macOS. Please run: brew install libomp")
            if "No such file or directory" in stderr:
                logger.error("Required input files not found. Feature engineering may have failed.")
        else:
            logger.info("Modeling Agent executed successfully")
            # Check if model files were created
            if os.path.exists("best_model.pkl"):
                logger.info("Successfully created: best_model.pkl")
            else:
                logger.warning("Expected file not created: best_model.pkl")
            if os.path.exists("scaler.pkl"):
                logger.info("Successfully created: scaler.pkl")
            else:
                logger.warning("Expected file not created: scaler.pkl")
        
        # Update state
        state["model_output"] = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        state["messages"].append(f"Modeling Agent completed. Output: {stdout[:500]}...")
        
        # Log the agent details
        state["logs"]["Modeling_Agent"] = {
            "prompt": model_prompt,
            "output_log": state["model_output"],
            "final_code": final_code
        }
        
        logger.info("Modeling Agent completed")
        return state
        
    except Exception as e:
        logger.error(f"Fatal error in Modeling Agent: {str(e)}")
        state["model_output"] = f"ERROR: {str(e)}"
        state["messages"].append(f"Modeling Agent failed: {str(e)}")
        return state

# Evaluation Agent
def evaluation_agent(state: GraphState) -> GraphState:
    """Evaluation Agent - Evaluates the final model on test set"""
    logger.info("=" * 60)
    logger.info("Starting Evaluation Agent")
    
    eval_prompt = """You are a model evaluation expert agent tasked with evaluating the final model on Microsoft (MSFT) stock test data.

Your task is to write a complete Python script called EVAL.py that:
1. Loads the test features from 'data/MSFT_test_features.csv'
2. CRITICAL: Load model and scaler using joblib (not pickle):
   - import joblib
   - model = joblib.load('best_model.pkl')
   - scaler = joblib.load('scaler.pkl') if os.path.exists('scaler.pkl') else None
3. Prepares the test data:
   - Parse Date column if present: test_data['Date'] = pd.to_datetime(test_data['Date'])
   - Extract target: y_test = test_data['Target_return']
   - Drop Date and Target_return: X_test = test_data.drop(columns=['Date', 'Target_return'], errors='ignore')
   - Apply scaler if it exists: if scaler: X_test = scaler.transform(X_test)
   - Handle any NaN values
4. Makes predictions on the test set
5. Calculates the RMSE (Root Mean Square Error):
   - from sklearn.metrics import mean_squared_error
   - rmse = np.sqrt(mean_squared_error(y_test, predictions))
6. Creates visualizations showing:
   - Predicted vs Actual scatter plot
   - Residuals distribution
   - Time series plot of predictions vs actuals
7. Saves visualizations as PNG files
8. CRITICAL: Write RMSE to 'MSFT_Score.txt' with EXACTLY this format:
   with open('MSFT_Score.txt', 'w') as f:
       f.write(f'RMSE: {rmse}\\n')
   (No extra text, just "RMSE: " followed by the number)
9. Prints detailed evaluation metrics and analysis

IMPORTANT:
- Use joblib.load(), not pickle.load()
- Ensure Date column is excluded from features
- Handle missing scaler file gracefully
- The MSFT_Score.txt file MUST be created with the exact format shown

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Evaluation Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Evaluation prompt")
        code = safe_llm_invoke(llm, [
            SystemMessage(content=eval_prompt),
            HumanMessage(content="Generate the EVAL.py script")
        ]).strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        
        logger.info("LLM generated EVAL.py script successfully")
        
        # Execute and debug the code
        stdout, stderr, final_code = execute_and_debug_code(code, "EVAL.py", llm, max_attempts=3)
        
        # Check for errors
        if stderr and "Warning" not in stderr:
            logger.error(f"Evaluation Agent encountered errors after debugging: {stderr}")
            if "No such file or directory" in stderr:
                logger.error("Required input files not found. Previous agents may have failed.")
        else:
            logger.info("Evaluation Agent executed successfully")
            # Check if MSFT_Score.txt was created
            if os.path.exists("MSFT_Score.txt"):
                logger.info("Successfully created: MSFT_Score.txt")
                with open("MSFT_Score.txt", "r") as f:
                    score = f.read().strip()
                    logger.info(f"Final score: {score}")
            else:
                logger.error("MSFT_Score.txt was not created!")
        
        # Update state
        state["eval_output"] = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        state["messages"].append(f"Evaluation Agent completed. Output: {stdout[:500]}...")
        
        # Log the agent details
        state["logs"]["Evaluation_Agent"] = {
            "prompt": eval_prompt,
            "output_log": state["eval_output"],
            "final_code": final_code
        }
        
        logger.info("Evaluation Agent completed")
        return state
        
    except Exception as e:
        logger.error(f"Fatal error in Evaluation Agent: {str(e)}")
        state["eval_output"] = f"ERROR: {str(e)}"
        state["messages"].append(f"Evaluation Agent failed: {str(e)}")
        return state

# Create the workflow
def create_workflow():
    """Create the multi-agent workflow using LangGraph"""
    
    # Create a new graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("eda", eda_agent)
    workflow.add_node("feature_engineering", feature_engineering_agent)
    workflow.add_node("modeling", modeling_agent)
    workflow.add_node("evaluation", evaluation_agent)
    
    # Add edges - sequential flow
    workflow.set_entry_point("eda")
    workflow.add_edge("eda", "feature_engineering")
    workflow.add_edge("feature_engineering", "modeling")
    workflow.add_edge("modeling", "evaluation")
    workflow.add_edge("evaluation", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def main():
    """Main function to run the multi-agent workflow"""
    
    logger.info("=" * 80)
    logger.info("Starting Multi-Agent MSFT Stock Prediction Workflow")
    logger.info("=" * 80)
    
    print("Starting Multi-Agent MSFT Stock Prediction Workflow...")
    print("=" * 60)
    print("Check logs directory for detailed execution logs")
    print("=" * 60)
    
    # Create initial state
    initial_state = {
        "messages": [],
        "eda_output": "",
        "feature_output": "",
        "model_output": "",
        "eval_output": "",
        "logs": {}
    }
    
    # Create and run the workflow
    app = create_workflow()
    
    try:
        logger.info("Starting workflow execution")
        # Run the workflow
        final_state = app.invoke(initial_state)
        
        # Save the submission log
        submission_log = final_state["logs"]
        
        logger.info("Adding script contents to submission log")
        # Add both original generated scripts and final debugged versions
        add_file_content(submission_log, "EDA.py", "EDA_Script")
        add_file_content(submission_log, "FEATURE.py", "FeatureEngineering_Script")
        add_file_content(submission_log, "MODEL.py", "Modeling_Script")
        add_file_content(submission_log, "EVAL.py", "Evaluation_Script")
        
        # Save submission log
        with open("submission_log.json", "w") as f:
            json.dump(submission_log, f, indent=2)
        
        logger.info("Workflow completed")
        logger.info("=" * 80)
        
        print("\n" + "=" * 60)
        print("Workflow completed!")
        print(f"Generated files: EDA.py, FEATURE.py, MODEL.py, EVAL.py")
        
        # Check if MSFT_Score.txt exists and display result
        if os.path.exists("MSFT_Score.txt"):
            with open("MSFT_Score.txt", "r") as f:
                score = f.read().strip()
            print(f"✓ MSFT_Score.txt created successfully: {score}")
        else:
            print("✗ MSFT_Score.txt was NOT created - check logs for errors")
            
        print(f"Submission log saved to: submission_log.json")
        print(f"Detailed logs saved to: logs/")
        
        # Print a summary of errors if any
        logger.info("Checking for errors in agent outputs")
        error_summary = []
        for agent_name in ["EDA_Agent", "FeatureEngineering_Agent", "Modeling_Agent", "Evaluation_Agent"]:
            if agent_name in submission_log:
                output_log = submission_log[agent_name].get("output_log", "")
                if "STDERR:" in output_log and output_log.split("STDERR:")[1].strip():
                    error_summary.append(f"{agent_name}: {output_log.split('STDERR:')[1].strip()[:200]}...")
        
        if error_summary:
            print("\n⚠️  Errors detected during execution:")
            for error in error_summary:
                print(f"  - {error}")
            print("\nPlease check the logs for detailed error information.")
        
    except Exception as e:
        logger.error(f"Fatal error in workflow: {str(e)}")
        print(f"Error in workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Check if API credentials are set
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_azure = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    
    if not has_openai and not has_azure:
        print("No API credentials found!")
        print("\nFor standard OpenAI:")
        print("export OPENAI_API_KEY='your-api-key'")
        print("\nFor Azure OpenAI:")
        print("export AZURE_OPENAI_API_KEY='your-azure-key'")
        print("export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com'")
        print("export AZURE_OPENAI_DEPLOYMENT_NAME='your-deployment-name'")
        print("export AZURE_OPENAI_API_VERSION='2023-05-15'")
        sys.exit(1)
    
    if has_azure:
        print("Using Azure OpenAI credentials")
    else:
        print("Using standard OpenAI credentials")
    
    main()
