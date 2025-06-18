#!/usr/bin/env python3
"""
Multi-Agent System for MSFT Stock Prediction using LangGraph
"""

import os
import json
import subprocess
import sys
from typing import Dict, Any, TypedDict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Import prompts from external file
from agent_prompts import (
    EDA_PROMPT,
    FEATURE_ENGINEERING_PROMPT,
    MODELING_PROMPT,
    EVALUATION_PROMPT
)

# Global variables for logging
submission_log = {
    "EDA_Agent": {"prompt": "", "output_log": ""},
    "FeatureEngineering_Agent": {"prompt": "", "output_log": ""},
    "Modeling_Agent": {"prompt": "", "output_log": ""},
    "Evaluation_Agent": {"prompt": "", "output_log": ""}
}

class AgentState(TypedDict):
    messages: list
    current_agent: str
    execution_status: str
    error_message: str
    scripts_generated: Dict[str, str]

def add_file_content(logs: dict, filename: str, log_key: str):
    """Helper function to add file content to logs"""
    try:
        with open(filename, "r") as f:
            content = f.read()
    except Exception as e:
        content = f"Error reading {filename}: {e}"
    logs[log_key] = content

@tool
def execute_python_script(script_path: str) -> str:
    """Execute a Python script and return the output"""
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        return f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Script execution timed out after 5 minutes"
    except Exception as e:
        return f"Error executing script: {str(e)}"

@tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file"""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"Error writing to {filename}: {str(e)}"

class MSFTAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        self.tools = [execute_python_script, write_file]
        
    def create_eda_agent(self, state: AgentState) -> AgentState:
        """EDA Agent: Generates EDA.py script for exploratory data analysis"""
        global submission_log
        
        submission_log["EDA_Agent"]["prompt"] = EDA_PROMPT
        
        messages = [SystemMessage(content=EDA_PROMPT)]
        response = self.llm.invoke(messages)
        
        # Extract Python code from response
        code_content = response.content
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].split("```")[0].strip()
        
        # Write the EDA script
        with open("EDA.py", "w") as f:
            f.write(code_content)
        
        # Execute the script
        execution_result = execute_python_script.invoke({"script_path": "EDA.py"})
        
        submission_log["EDA_Agent"]["output_log"] = execution_result
        
        state["current_agent"] = "FeatureEngineering_Agent"
        state["scripts_generated"]["EDA"] = code_content
        state["execution_status"] = "EDA completed"
        
        return state
    
    def create_feature_engineering_agent(self, state: AgentState) -> AgentState:
        """Feature Engineering Agent: Generates FEATURE.py script"""
        global submission_log
        
        submission_log["FeatureEngineering_Agent"]["prompt"] = FEATURE_ENGINEERING_PROMPT
        
        messages = [SystemMessage(content=FEATURE_ENGINEERING_PROMPT)]
        response = self.llm.invoke(messages)
        
        # Extract Python code from response
        code_content = response.content
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].split("```")[0].strip()
        
        # Write the FEATURE script
        with open("FEATURE.py", "w") as f:
            f.write(code_content)
        
        # Execute the script
        execution_result = execute_python_script.invoke({"script_path": "FEATURE.py"})
        
        submission_log["FeatureEngineering_Agent"]["output_log"] = execution_result
        
        state["current_agent"] = "Modeling_Agent"
        state["scripts_generated"]["FEATURE"] = code_content
        state["execution_status"] = "Feature Engineering completed"
        
        return state
    
    def create_modeling_agent(self, state: AgentState) -> AgentState:
        """Modeling Agent: Generates MODEL.py script"""
        global submission_log
        
        submission_log["Modeling_Agent"]["prompt"] = MODELING_PROMPT
        
        messages = [SystemMessage(content=MODELING_PROMPT)]
        response = self.llm.invoke(messages)
        
        # Extract Python code from response
        code_content = response.content
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].split("```")[0].strip()
        
        # Write the MODEL script
        with open("MODEL.py", "w") as f:
            f.write(code_content)
        
        # Execute the script
        execution_result = execute_python_script.invoke({"script_path": "MODEL.py"})
        
        submission_log["Modeling_Agent"]["output_log"] = execution_result
        
        state["current_agent"] = "Evaluation_Agent"
        state["scripts_generated"]["MODEL"] = code_content
        state["execution_status"] = "Modeling completed"
        
        return state
    
    def create_evaluation_agent(self, state: AgentState) -> AgentState:
        """Evaluation Agent: Generates EVAL.py script"""
        global submission_log
        
        submission_log["Evaluation_Agent"]["prompt"] = EVALUATION_PROMPT
        
        messages = [SystemMessage(content=EVALUATION_PROMPT)]
        response = self.llm.invoke(messages)
        
        # Extract Python code from response
        code_content = response.content
        if "```python" in code_content:
            code_content = code_content.split("```python")[1].split("```")[0].strip()
        elif "```" in code_content:
            code_content = code_content.split("```")[1].split("```")[0].strip()
        
        # Write the EVAL script
        with open("EVAL.py", "w") as f:
            f.write(code_content)
        
        # Execute the script
        execution_result = execute_python_script.invoke({"script_path": "EVAL.py"})
        
        submission_log["Evaluation_Agent"]["output_log"] = execution_result
        
        state["current_agent"] = "completed"
        state["scripts_generated"]["EVAL"] = code_content
        state["execution_status"] = "Evaluation completed"
        
        return state

def create_workflow():
    """Create the LangGraph workflow"""
    
    # Initialize the workflow
    workflow = StateGraph(AgentState)
    
    # Create agent instance
    agent_system = MSFTAnalysisAgent()
    
    # Add nodes
    workflow.add_node("eda_agent", agent_system.create_eda_agent)
    workflow.add_node("feature_agent", agent_system.create_feature_engineering_agent)
    workflow.add_node("modeling_agent", agent_system.create_modeling_agent)
    workflow.add_node("evaluation_agent", agent_system.create_evaluation_agent)
    
    # Define the flow
    workflow.set_entry_point("eda_agent")
    workflow.add_edge("eda_agent", "feature_agent")
    workflow.add_edge("feature_agent", "modeling_agent")
    workflow.add_edge("modeling_agent", "evaluation_agent")
    workflow.add_edge("evaluation_agent", END)
    
    return workflow.compile()

def main():
    """Main execution function"""
    print("Starting MSFT Stock Prediction Multi-Agent System...")
    
    # Create and run the workflow
    app = create_workflow()
    
    # Initial state
    initial_state = {
        "messages": [],
        "current_agent": "EDA_Agent",
        "execution_status": "Starting",
        "error_message": "",
        "scripts_generated": {}
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Add script contents to submission log
    script_logs = {}
    add_file_content(script_logs, "EDA.py", "EDA_Script")
    add_file_content(script_logs, "FEATURE.py", "FeatureEngineering_Script") 
    add_file_content(script_logs, "MODEL.py", "Modeling_Script")
    add_file_content(script_logs, "EVAL.py", "Evaluation_Script")
    
    # Save submission log
    with open("submission_log.json", "w") as f:
        json.dump(submission_log, f, indent=2)
    
    print("Multi-Agent System execution completed!")
    print("Generated files:")
    print("- EDA.py")
    print("- FEATURE.py") 
    print("- MODEL.py")
    print("- EVAL.py")
    print("- submission_log.json")
    print("- MSFT_Score.txt")

if __name__ == "__main__":
    main()

"""
Agent prompts for MSFT Stock Prediction System
"""

EDA_PROMPT = """You are an expert data scientist specializing in Exploratory Data Analysis (EDA). Your task is to generate a complete Python script called EDA.py that will:

1. Download Microsoft (MSFT) stock data from 2016-01-01 to 2024-12-30 if not available
2. Load and examine the OHLC (Open, High, Low, Close) and Volume data
3. Calculate the target variable: Target_return = ln(Close[t+1]) - ln(Close[t])
4. Create train/validation/test splits chronologically
5. Perform comprehensive EDA including:
   - Data shape, types, missing values analysis
   - Statistical summaries
   - Time series plots of price and volume
   - Distribution analysis of returns
   - Correlation analysis
   - Volatility analysis
   - Save visualizations as PNG files

The script must be self-contained and executable. Create the data/ folder if it doesn't exist.
IMPORTANT: The existing data/MSFT.csv has multi-level headers. Load it with:
pd.read_csv('data/MSFT.csv', header=[0,1], index_col=0)
Save splits maintaining the multi-level header structure.

Generate ONLY the Python code for EDA.py - no explanations or markdown."""

FEATURE_ENGINEERING_PROMPT = """You are an expert quantitative analyst. Generate a Python script called FEATURE.py that creates advanced features for stock prediction:

1. Load data with multi-level headers:
   train = pd.read_csv('data/train.csv', header=[0,1], index_col=0)
   val = pd.read_csv('data/val.csv', header=[0,1], index_col=0)
   test = pd.read_csv('data/test.csv', header=[0,1], index_col=0)
   # Flatten columns: train.columns = [col[0] for col in train.columns]

2. Create 150+ features using pandas only (no talib):
   a) Price features:
      - SMA: periods = [5, 10, 15, 20, 30, 50, 100, 200]
      - EMA: periods = [5, 10, 12, 20, 26, 50]
      - Price/SMA ratios for each period
      - Price momentum: Close/Close.shift(n) for n in [1,2,3,5,10,20]
      - Log returns: np.log(Close/Close.shift(n)) for multiple n
      
   b) Volatility features:
      - Rolling std: periods = [5, 10, 20, 30, 50, 100]
      - Parkinson: np.sqrt((1/(4*np.log(2))) * (np.log(High/Low)**2).rolling(n).mean())
      - Garman-Klass estimator
      - Yang-Zhang estimator
      - Close-to-close, high-low, and overnight volatilities
      
   c) Volume features:
      - Volume SMA: periods = [5, 10, 20, 50]
      - Volume/Volume_SMA ratios
      - Price*Volume
      - Volume rate of change
      - On Balance Volume (OBV) approximation
      
   d) Technical indicators (pandas implementation):
      - RSI for periods [7, 14, 21, 28]
      - Bollinger Bands: upper, lower, width, %B
      - MACD: EMA12 - EMA26, signal line
      - Stochastic %K and %D
      - Williams %R
      - Money Flow Index approximation
      
   e) Market microstructure:
      - High-Low spread: (High-Low)/Close
      - Close-Open spread: (Close-Open)/Open
      - Intraday volatility: (High-Low)/(High+Low)
      - Overnight gap: Open/Close.shift(1)
      
   f) Lag features:
      - Lagged returns: 1 to 30 days
      - Lagged volume: 1 to 20 days
      - Lagged volatility: 1 to 20 days
      - Moving correlations between price and volume
      
   g) Statistical features:
      - Rolling skewness: periods = [10, 20, 50]
      - Rolling kurtosis: periods = [10, 20, 50]
      - Rolling min/max over various windows
      - Z-scores: (Close - SMA) / rolling_std
      
   h) Time-based features:
      - Day of week (0-4 for Mon-Fri)
      - Day of month
      - Month of year
      - Quarter
      - Is month start/end
      - Days to month end

3. Handle data quality:
   - Replace inf/-inf with NaN
   - Forward fill then backward fill NaN
   - Clip extreme values to 99.9 percentile

4. Save with same multi-level structure as input

Generate ONLY Python code - no explanations or markdown."""

MODELING_PROMPT = """You are an expert machine learning engineer specializing in financial time series forecasting. Your task is to generate a complete Python script called MODEL.py that will:

1. Load data from data/train.csv and data/val.csv with multi-level headers using:
   train = pd.read_csv('data/train.csv', header=[0,1], index_col=0)
   val = pd.read_csv('data/val.csv', header=[0,1], index_col=0)
   # Flatten columns: train.columns = [col[0] for col in train.columns]

2. Implement a sophisticated ensemble approach:
   a) Advanced XGBoost with extensive tuning:
      - n_estimators: [500, 1000, 1500, 2000]
      - max_depth: [3, 4, 5, 6, 7, 8]
      - learning_rate: [0.005, 0.01, 0.015, 0.02, 0.03]
      - subsample: [0.6, 0.7, 0.8, 0.9]
      - colsample_bytree: [0.6, 0.7, 0.8, 0.9]
      - gamma: [0, 0.1, 0.2, 0.3]
      - reg_alpha: [0, 0.01, 0.1, 1]
      - reg_lambda: [0, 0.01, 0.1, 1]
   
   b) LightGBM with advanced params:
      - Similar extensive hyperparameter search
      - Use 'dart' boosting type for better generalization
      - Enable categorical feature support
   
   c) CatBoost (if available, else skip):
      - Automatic categorical feature handling
      - Ordered boosting for time series
   
   d) Neural Network (using sklearn MLPRegressor):
      - Multiple architectures: (100,50), (200,100,50), (300,200,100)
      - Different activation functions
      - Early stopping and adaptive learning rate
   
   e) Ridge/Lasso with polynomial features:
      - Create polynomial features (degree 2)
      - Use both Ridge and Lasso with CV

3. Advanced Ensemble Techniques:
   - Train a meta-model (Ridge) on validation predictions
   - Use optimal weighted averaging based on validation performance
   - Try stacking with cross-validation
   - Implement voting regressor with soft voting

4. Feature Engineering within the script:
   - Add rolling statistics (5, 10, 20, 50 periods)
   - Price momentum features
   - Technical indicators using pandas only
   - Interaction features between volume and price changes

5. Robust validation:
   - Use TimeSeriesSplit for cross-validation
   - Implement proper walk-forward validation
   - Track performance metrics for each fold

6. Save the best model/ensemble with joblib

CRITICAL: The current RMSE is 0.209936. You MUST beat this by:
- Using ALL features in the data (don't drop any)
- Extensive hyperparameter optimization
- Sophisticated ensembling
- Proper time series validation

IMPORTANT: DO NOT use tensorflow/keras. Use only sklearn, xgboost, lightgbm, and catboost.
Handle any import errors gracefully - if a library isn't available, skip that model.

Generate ONLY the Python code for MODEL.py - no explanations or markdown."""

EVALUATION_PROMPT = """You are an expert model evaluation specialist. Generate a Python script called EVAL.py:

1. Load test data with multi-level headers:
   test = pd.read_csv('data/test.csv', header=[0,1], index_col=0)
   test.columns = [col[0] for col in test.columns]
   
2. Load the saved model:
   import joblib
   model = joblib.load('best_model.pkl')
   
3. Prepare test data:
   - Separate features and target
   - X_test = test.drop('Target_return', axis=1)
   - y_test = test['Target_return']
   - Handle any feature engineering if saved scaler exists
   
4. Generate predictions:
   - If model is a dict (ensemble), handle appropriately
   - Otherwise: predictions = model.predict(X_test)
   
5. Calculate and save RMSE:
   from sklearn.metrics import mean_squared_error
   import numpy as np
   rmse = np.sqrt(mean_squared_error(y_test, predictions))
   with open('MSFT_Score.txt', 'w') as f:
       f.write(f'RMSE: {rmse:.6f}')
   
6. Additional metrics:
   - MAE
   - Directional accuracy: (np.sign(y_test[1:]) == np.sign(predictions[1:])).mean()
   - R-squared score
   - Mean percentage error
   
7. Create simple plot:
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.scatter(y_test, predictions, alpha=0.5)
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
   plt.xlabel('Actual')
   plt.ylabel('Predicted')
   plt.title('Actual vs Predicted Returns')
   plt.savefig('evaluation_plot.png')
   
8. Print results

Generate ONLY Python code - no explanations."""
