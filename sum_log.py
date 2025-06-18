"""
Enhanced submission log saver that creates a more readable format
"""

import json
import os

def save_enhanced_submission_log(submission_log):
    """Save submission log with better formatting"""
    
    # Create a logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Save each agent's prompt to a separate file for readability
    for agent_name, agent_data in submission_log.items():
        # Save prompt to a separate text file
        prompt_filename = f"logs/{agent_name}_prompt.txt"
        with open(prompt_filename, 'w') as f:
            f.write(agent_data.get('prompt', ''))
        
        # Save output log to a separate file
        output_filename = f"logs/{agent_name}_output.txt"
        with open(output_filename, 'w') as f:
            f.write(agent_data.get('output_log', ''))
    
    # Create a summary JSON with file references
    summary = {}
    for agent_name in submission_log.keys():
        summary[agent_name] = {
            "prompt_file": f"logs/{agent_name}_prompt.txt",
            "output_file": f"logs/{agent_name}_output.txt"
        }
    
    # Save the summary
    with open("logs/submission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Also save the original format for compatibility
    with open("submission_log.json", "w") as f:
        json.dump(submission_log, f, indent=2, ensure_ascii=False)
    
    print("Submission logs saved in 'logs/' directory")
    print("- Individual prompt files: logs/*_prompt.txt")
    print("- Individual output files: logs/*_output.txt")
    print("- Summary file: logs/submission_summary.json")
    print("- Original format: submission_log.json")
