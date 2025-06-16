
#!/usr/bin/env python3
"""
Minimal script to reproduce and fix the SystemMessage error with Langchain 0.3.25
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing Langchain 0.3.25 with Langgraph 0.4.8")
print("=" * 60)

# Define a simple state
class GraphState(TypedDict):
    messages: List[str]
    result: str

# Test 1: Direct LLM invocation (this likely causes the error)
def test_direct_invocation():
    print("\nTest 1: Direct LLM invocation")
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say hello")
        ])
        print(f"Success! Response: {response.content}")
        return True
    except AttributeError as e:
        print(f"ERROR: {e}")
        return False

# Test 2: Safe invocation with proper handling
def safe_llm_invoke(llm, messages):
    """Safe wrapper that handles different Langchain versions"""
    try:
        response = llm.invoke(messages)
        
        # Handle different response types
        if isinstance(response, AIMessage):
            return response.content
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except AttributeError as e:
        # If standard invocation fails, try with message dictionaries
        print(f"Standard invocation failed: {e}")
        print("Trying alternative method...")
        
        # Convert messages to dictionaries
        message_dicts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                message_dicts.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                message_dicts.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_dicts.append({"role": "assistant", "content": msg.content})
        
        # Try with dictionaries
        response = llm.invoke(message_dicts)
        
        if isinstance(response, AIMessage):
            return response.content
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)

def test_safe_invocation():
    print("\nTest 2: Safe LLM invocation")
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        result = safe_llm_invoke(llm, [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say hello")
        ])
        print(f"Success! Response: {result}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

# Test 3: LangGraph integration with safe invocation
def node_with_llm(state: GraphState) -> GraphState:
    """Example node that uses LLM safely"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Use safe invocation
    result = safe_llm_invoke(llm, [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Generate a greeting")
    ])
    
    state["result"] = result
    state["messages"].append(f"LLM said: {result}")
    return state

def test_langgraph_integration():
    print("\nTest 3: LangGraph integration")
    try:
        # Create workflow
        workflow = StateGraph(GraphState)
        workflow.add_node("llm_node", node_with_llm)
        workflow.set_entry_point("llm_node")
        workflow.add_edge("llm_node", END)
        
        # Compile and run
        app = workflow.compile()
        initial_state = {"messages": [], "result": ""}
        final_state = app.invoke(initial_state)
        
        print(f"Success! Final result: {final_state['result']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main execution
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run tests
    test1_passed = test_direct_invocation()
    test2_passed = test_safe_invocation()
    test3_passed = test_langgraph_integration()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Test 1 (Direct invocation): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Safe invocation): {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (LangGraph integration): {'PASSED' if test3_passed else 'FAILED'}")
    
    if test2_passed and test3_passed:
        print("\n✅ Solution: Use the safe_llm_invoke wrapper function!")
        print("This handles compatibility issues between different Langchain versions.")
