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

# Activate the conda environment
os.environ['PATH'] = f"/Users/dimitermilushev/miniforge3/envs/msft_agents/bin:{os.environ['PATH']}"
sys.path.insert(0, '/Users/dimitermilushev/miniforge3/envs/msft_agents/lib/python3.10/site-packages')

# Define the state for our workflow
class GraphState(TypedDict):
    """State passed between agents"""
    messages: List[str]
    eda_output: str
    feature_output: str 
    model_output: str
    eval_output: str
    logs: Dict[str, Any]

# Initialize the LLM
def get_llm():
    """Get the LLM instance"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    logger.debug("Successfully initialized LLM with OpenAI API")
    return ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

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
                ["/Users/dimitermilushev/miniforge3/envs/msft_agents/bin/python", filename],
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
3. Missing column handling when dropping Date/Target_return columns
4. Pickle vs joblib consistency

Return ONLY the corrected Python code, nothing else.
"""
                
                try:
                    debug_response = llm.invoke([
                        SystemMessage(content=debug_prompt),
                        HumanMessage(content="Fix the code")
                    ])
                    
                    # Handle the response properly for newer Langchain versions
                    # In Langchain 0.3.x, response is an AIMessage object
                    if isinstance(debug_response, AIMessage):
                        corrected_code = debug_response.content.strip()
                    elif hasattr(debug_response, 'content'):
                        corrected_code = debug_response.content.strip()
                    else:
                        corrected_code = str(debug_response).strip()
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
2. Performs comprehensive exploratory data analysis including:
   - Data shape and basic info
   - Statistical summary of all columns
   - Check for missing values
   - Analyze the distribution of the target variable (Target_return)
   - Create visualizations for price trends, volume, and returns
   - Analyze correlations between features
   - Check for any data quality issues
3. Save all visualizations as PNG files with descriptive names
4. Print key insights and findings

The data contains: Date, Open, High, Low, Close, Volume, Target_return (log return for next day)

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.
The script should be production-ready and well-commented.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for EDA Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with EDA prompt")
        response = llm.invoke([
            SystemMessage(content=eda_prompt),
            HumanMessage(content="Generate the EDA.py script")
        ])
        
        # Handle response for newer Langchain versions
        # In Langchain 0.3.x, response is an AIMessage object
        if isinstance(response, AIMessage):
            code = response.content.strip()
        elif hasattr(response, 'content'):
            code = response.content.strip()
        else:
            code = str(response).strip()
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
2. Creates meaningful features for predicting next-day log returns, including:
   - Technical indicators (moving averages, RSI, MACD, Bollinger Bands, etc.)
   - Price-based features (returns, log returns, price ratios)
   - Volume-based features (volume moving averages, volume ratios)
   - Volatility features (rolling standard deviation, ATR)
   - Momentum features
   - Any other relevant financial features
3. Handles the temporal nature of the data properly (no look-ahead bias)
4. Saves the engineered features for each dataset as:
   - 'data/MSFT_train_features.csv'
   - 'data/MSFT_val_features.csv'  
   - 'data/MSFT_test_features.csv'
5. Prints the list of created features and their importance/relevance

The target variable is 'Target_return' (log return for next day).

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.
Ensure no data leakage between train/val/test sets.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Feature Engineering Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Feature Engineering prompt")
        response = llm.invoke([
            SystemMessage(content=feature_prompt),
            HumanMessage(content="Generate the FEATURE.py script")
        ])
        
        # Handle response for newer Langchain versions
        # In Langchain 0.3.x, response is an AIMessage object
        if isinstance(response, AIMessage):
            code = response.content.strip()
        elif hasattr(response, 'content'):
            code = response.content.strip()
        else:
            code = str(response).strip()
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
2. Prepares the data for modeling (handle NaN values, scale features if needed)
3. Trains multiple models to predict 'Target_return', including:
   - Linear Regression (baseline)
   - Random Forest
   - XGBoost
   - LightGBM
   - Any other suitable models
4. Performs hyperparameter tuning using the validation set
5. Evaluates each model on the validation set using appropriate metrics (RMSE, MAE, R2)
6. Selects the best model based on validation performance
7. Saves the best model as 'best_model.pkl' using joblib
8. Saves the scaler (if used) as 'scaler.pkl'
9. Prints model performance comparisons and feature importances

Important: Do NOT use the test set for training or model selection. Only train/val sets.

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Modeling Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Modeling prompt")
        response = llm.invoke([
            SystemMessage(content=model_prompt),
            HumanMessage(content="Generate the MODEL.py script")
        ])
        
        # Handle response for newer Langchain versions
        # In Langchain 0.3.x, response is an AIMessage object
        if isinstance(response, AIMessage):
            code = response.content.strip()
        elif hasattr(response, 'content'):
            code = response.content.strip()
        else:
            code = str(response).strip()
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
2. Loads the saved best model from 'best_model.pkl'
3. Loads the scaler from 'scaler.pkl' (if it exists)
4. Prepares the test data (apply same preprocessing as training)
5. Makes predictions on the test set
6. Calculates the RMSE (Root Mean Square Error) between predictions and actual Target_return values
7. Creates visualizations showing:
   - Predicted vs Actual scatter plot
   - Residuals distribution
   - Time series plot of predictions vs actuals
8. Saves visualizations as PNG files
9. Writes the RMSE score to 'MSFT_Score.txt' with EXACTLY this format:
   RMSE: <value>
   (where <value> is the float RMSE score, nothing else)
10. Prints detailed evaluation metrics and analysis

Write a complete, executable Python script. Include all necessary imports. Handle any potential errors gracefully.

Output only the Python code, nothing else."""

    try:
        logger.debug("Initializing LLM for Evaluation Agent")
        llm = get_llm()
        
        logger.debug("Invoking LLM with Evaluation prompt")
        response = llm.invoke([
            SystemMessage(content=eval_prompt),
            HumanMessage(content="Generate the EVAL.py script")
        ])
        
        # Handle response for newer Langchain versions
        # In Langchain 0.3.x, response is an AIMessage object
        if isinstance(response, AIMessage):
            code = response.content.strip()
        elif hasattr(response, 'content'):
            code = response.content.strip()
        else:
            code = str(response).strip()
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
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)
    
    main()
