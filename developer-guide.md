# MRMC Agent Bot - Developer Guide

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repo-url>
cd mrm

# Install dependencies
pip install -r requirements.txt
```

### Running the System

#### Test with Mock LLM (No API Key Required)
```bash
# Test full system
python -m mrmc_agent.cli test-full-system . --mock

# Test individual tools
python -m mrmc_agent.cli test-tools .

# Test specific agent
python -m mrmc_agent.cli test-methodology . --mock
```

#### Production Mode with Real LLM
```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Run full analysis
python -m mrmc_agent.cli test-full-system /path/to/your/ml/project
```

## Architecture Overview

### Core Components

```
mrmc_agent/
├── core/                   # Core framework
│   ├── base_agent.py      # Base agent class
│   ├── orchestrator.py    # Coordinates all agents
│   ├── tool_manager.py    # Manages analysis tools
│   └── llm_service.py     # LLM reasoning engine
│
├── agents/                 # Specialized MRMC agents
│   ├── purpose_agent.py   # Section 1: Model Purpose
│   ├── inputs_agent.py    # Section 2: Model Inputs
│   ├── methodology_agent.py # Section 3: Model Methodology
│   ├── implementation_agent.py # Section 4: Implementation
│   └── usage_agent.py     # Section 5: Model Usage
│
├── tools/                  # Analysis tools
│   ├── ast_analyzer.py    # Python AST analysis
│   ├── code_search.py     # Code pattern search
│   └── config_reader.py   # Configuration parsing
│
└── cli.py                 # Command-line interface
```

## How It Works

### 1. Agent Initialization
```python
# Each agent needs tools and LLM service
tool_manager = ToolManager()
tool_manager.register_tool(PythonASTAnalyzer())
tool_manager.register_tool(CodeSearchTool())
tool_manager.register_tool(ConfigReaderTool())

llm_service = LLMService()  # Or MockLLMService for testing

# Create specialized agents
purpose_agent = PurposeAgent(tool_manager, llm_service)
inputs_agent = InputsAgent(tool_manager, llm_service)
# ... etc
```

### 2. Orchestration
```python
# Orchestrator coordinates all agents
orchestrator = OrchestratorAgent(tool_manager)
orchestrator.register_agent(purpose_agent)
orchestrator.register_agent(inputs_agent)
# ... register all agents

# Generate complete MRMC document
result = await orchestrator.analyze(repo_path, context={})
document = result.data  # MRMCDocument object
```

### 3. Agent Workflow
Each agent follows this pattern:

```python
class SpecializedAgent(BaseAgent):
    async def analyze(self, repo_path: str, context: Dict) -> AgentResult:
        # 1. Gather evidence using tools
        tool_results = await self._gather_evidence(repo_path)
        
        # 2. Use LLM to reason about findings
        llm_response = await self.llm_service.reason_about_code(
            section_context=self._get_section_context(),
            tool_results=tool_results,
            specific_prompt=self._get_analysis_prompt()
        )
        
        # 3. Return structured results
        return AgentResult(
            success=True,
            data=llm_response.content,
            confidence=llm_response.confidence,
            evidence=[...]
        )
```

## Creating a New Agent

### Step 1: Define the Agent Class
```python
from typing import Dict, Any
from ..core.base_agent import BaseAgent, AgentResult
from ..core.tool_manager import ToolManager
from ..core.llm_service import LLMService

class MyCustomAgent(BaseAgent):
    def __init__(self, tool_manager: ToolManager, llm_service: LLMService):
        super().__init__("my_custom", "Description of what this agent does")
        self.tool_manager = tool_manager
        self.llm_service = llm_service
```

### Step 2: Implement Required Methods
```python
    async def analyze(self, repo_path: str, context: Dict[str, Any]) -> AgentResult:
        # Your analysis logic here
        pass
    
    def get_required_tools(self) -> list:
        return ["code_search", "config_reader", "ast_analysis"]
    
    async def _execute_task_impl(self, task) -> AgentResult:
        return await self.analyze(task.description, {})
    
    async def _generate_section_content(self, data: Any) -> str:
        return str(data)
```

### Step 3: Add Evidence Gathering
```python
    async def _gather_evidence(self, repo_path: str) -> Dict[str, Any]:
        results = {}
        
        # Use tools to gather evidence
        if self.tool_manager.has_tool("code_search"):
            search_tool = self.tool_manager.get_tool("code_search")
            search_result = await search_tool.execute(
                repo_path=repo_path,
                patterns=["your", "search", "patterns"],
                file_extensions=[".py"]
            )
            results["search"] = search_result.data if search_result.success else {}
        
        return results
```

### Step 4: Add LLM Prompts
```python
    def _get_section_context(self) -> str:
        return """
        Explain what this MRMC section requires...
        """
    
    def _get_analysis_prompt(self) -> str:
        return """
        Specific instructions for the LLM to analyze this section...
        """
```

## Creating a New Tool

### Step 1: Define the Tool Class
```python
from ..core.tool_manager import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="What this tool does"
        )
```

### Step 2: Implement Execute Method
```python
    async def execute(self, **kwargs) -> ToolResult:
        try:
            # Your tool logic here
            repo_path = kwargs.get("repo_path")
            
            # Do analysis...
            results = analyze_something(repo_path)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={"files_analyzed": 10}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
```

## Testing

### Unit Tests
```python
# Test individual agent
import pytest
from mrmc_agent.agents.purpose_agent import PurposeAgent

@pytest.mark.asyncio
async def test_purpose_agent():
    agent = PurposeAgent(mock_tool_manager, mock_llm_service)
    result = await agent.analyze("/test/repo", {})
    assert result.success
    assert result.confidence > 0.5
```

### Integration Tests
```bash
# Run with mock services
python -m mrmc_agent.cli test-full-system ./test-repos/sample-ml-project --mock

# Validate output
assert "5 MRMC sections" in output
assert all(section in output for section in ["Purpose", "Inputs", "Methodology", "Implementation", "Usage"])
```

## Configuration

### Environment Variables
```bash
# Required for production
export OPENAI_API_KEY=your-key-here

# Optional
export MRMC_LOG_LEVEL=DEBUG
export MRMC_CACHE_DIR=/tmp/mrmc-cache
export MRMC_TIMEOUT_SECONDS=60
```

### Configuration Files
```yaml
# config/mrmc_config.yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.3

agents:
  parallel_execution: true
  timeout_per_agent: 30

tools:
  cache_enabled: true
  cache_ttl: 300
```

## Performance Optimization

### 1. Parallel Agent Execution
The orchestrator runs all agents in parallel by default:
```python
# All 5 agents analyze simultaneously
results = await asyncio.gather(*[
    agent.analyze(repo_path, context) 
    for agent in agents
])
```

### 2. Tool Result Caching
Tools cache results to avoid redundant analysis:
```python
# Second call returns cached result
result1 = await tool.execute(repo_path="/path")  # Executes
result2 = await tool.execute(repo_path="/path")  # Returns cached
```

### 3. LLM Token Optimization
- Use specific, focused prompts
- Limit context to relevant information
- Configure max_tokens appropriately

## Troubleshooting

### Common Issues

#### 1. "Agent missing required tools"
**Problem**: Tools are registered with different names than agents expect
**Solution**: Ensure tools are registered with correct names
```python
# Tools register with these names:
tool_manager.register_tool(PythonASTAnalyzer())  # → "ast_analysis" 
tool_manager.register_tool(CodeSearchTool())     # → "code_search"
tool_manager.register_tool(ConfigReaderTool())   # → "config_reader"

# Agents expect these exact names in get_required_tools()
def get_required_tools(self) -> list:
    return ["code_search", "config_reader", "ast_analysis"]  # Must match exactly
```

#### 2. "AgentResult() got unexpected keyword argument"  
**Problem**: AgentResult doesn't have all required fields
**Solution**: Check AgentResult dataclass includes all fields used
```python
return AgentResult(
    success=True,
    data=content,
    confidence=0.8,
    evidence=["evidence1"],
    agent_name=self.name  # This field was recently added
)
```

#### 3. "RuntimeWarning: coroutine was never awaited"
**Problem**: Tool execution methods need proper await handling
**Solution**: Ensure all tool calls use proper async/await
```python
# Correct async usage
result = await tool_manager.execute_tool("ast_analysis", repo_path=path)

# Incorrect - missing await
result = tool_manager.execute_tool("ast_analysis", repo_path=path)
```

#### 4. "LLM timeout"
**Solution**: Increase timeout or use mock mode for testing
```bash
export MRMC_TIMEOUT_SECONDS=120
```

#### 5. "No sections generated" or "0.0% confidence"
**Problem**: In mock mode, this is expected behavior
**Solution**: 
- For testing: This is normal with `--mock` flag
- For production: Set `OPENAI_API_KEY` environment variable
- Check agent logs for specific failures:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### AgentResult  
```python
@dataclass
class AgentResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = None
    metadata: Dict[str, Any] = None
    agent_name: Optional[str] = None  # Added for tracking which agent generated result
```

### ToolResult
```python
@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str]
    execution_time: float
    metadata: Dict[str, Any]
```

### LLMResponse
```python
@dataclass
class LLMResponse:
    content: str
    confidence: float
    reasoning_steps: List[str]
    tokens_used: int
    success: bool
    error: Optional[str]
```

## Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to all public methods
- Write tests for new features

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Update documentation
6. Submit PR with clear description

## License

[Your License Here]

## Support

For issues or questions:
- GitHub Issues: [repo-url]/issues
- Documentation: See `/docs` folder
- Email: support@example.com