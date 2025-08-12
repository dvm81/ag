# MRMC Agent Bot: Visual Diagrams & Workflows

## 1. High-Level System Overview

```mermaid
graph TB
    subgraph "Input Sources"
        DEV[Developer]
        CI[CI/CD Pipeline]
        SCH[Scheduler]
    end
    
    subgraph "Entry Points"
        VSC[VS Code Extension]
        CLI[Command Line]
        API[REST API]
        GHA[GitHub Actions]
    end
    
    subgraph "MRMC Agent Brain"
        ORC{Orchestrator}
        AG1[Purpose Agent]
        AG2[Inputs Agent]
        AG3[Methodology Agent]
        AG4[Implementation Agent]
        AG5[Usage Agent]
    end
    
    subgraph "Knowledge Tools"
        T1[(Code Search)]
        T2[(AST Parser)]
        T3[(Config Reader)]
        T4[(Git History)]
        T5[(Test Analyzer)]
        T6[(Dependency Scanner)]
    end
    
    subgraph "Outputs"
        DOC[MRMC Document]
        VAL[Validation Report]
        AUDIT[Audit Log]
    end
    
    DEV --> VSC
    CI --> GHA
    SCH --> API
    
    VSC & CLI & API & GHA --> ORC
    
    ORC --> AG1 & AG2 & AG3 & AG4 & AG5
    
    AG1 & AG2 & AG3 & AG4 & AG5 -.-> T1 & T2 & T3 & T4 & T5 & T6
    
    ORC --> DOC
    DOC --> VAL
    DOC --> AUDIT
    
    style ORC fill:#f9f,stroke:#333,stroke-width:4px
    style DOC fill:#9f9,stroke:#333,stroke-width:2px
```

## 2. Agent Communication Flow

```mermaid
sequenceDiagram
    participant User
    participant Orchestrator
    participant PurposeAgent
    participant InputsAgent
    participant MethodologyAgent
    participant ImplementationAgent
    participant UsageAgent
    participant Tools
    participant LLM
    
    User->>Orchestrator: Generate MRMC for repo
    Orchestrator->>Orchestrator: Analyze repo structure
    
    par Parallel Execution
        Orchestrator->>PurposeAgent: Find model purpose
        PurposeAgent->>Tools: Search README, main()
        Tools-->>PurposeAgent: Entry points, docs
        PurposeAgent->>LLM: Interpret purpose
        LLM-->>PurposeAgent: Business context
        PurposeAgent-->>Orchestrator: Section 1 complete
    and
        Orchestrator->>InputsAgent: Find data sources
        InputsAgent->>Tools: Search data loads
        Tools-->>InputsAgent: Files, DBs, APIs
        InputsAgent->>LLM: Describe inputs
        LLM-->>InputsAgent: Input documentation
        InputsAgent-->>Orchestrator: Section 2 complete
    and
        Orchestrator->>MethodologyAgent: Extract methodology
        MethodologyAgent->>Tools: Find algorithms
        Tools-->>MethodologyAgent: Model classes
        MethodologyAgent->>LLM: Explain methodology
        LLM-->>MethodologyAgent: Algorithm description
        MethodologyAgent-->>Orchestrator: Section 3 complete
    end
    
    Orchestrator->>Orchestrator: Combine sections
    Orchestrator->>User: Complete MRMC document
```

## 3. Tool Usage Pattern

```mermaid
graph LR
    subgraph "Agent Request"
        A[Agent] -->|"Find all model classes"| TM[Tool Manager]
    end
    
    subgraph "Tool Execution"
        TM --> T1[Code Search]
        T1 -->|"*.py files"| R1[Search Results]
        
        TM --> T2[AST Parser]
        T2 -->|"Parse classes"| R2[Class Definitions]
        
        TM --> T3[Import Analyzer]
        T3 -->|"ML libraries"| R3[Dependencies]
    end
    
    subgraph "Result Processing"
        R1 & R2 & R3 --> AGG[Aggregator]
        AGG -->|"Structured data"| A
    end
    
    style TM fill:#ff9,stroke:#333,stroke-width:2px
```

## 4. Document Generation Pipeline

```mermaid
flowchart TB
    START([Start]) --> SCAN[Repository Scan]
    SCAN --> DISPATCH{Dispatch to Agents}
    
    DISPATCH --> PA[Purpose Analysis]
    DISPATCH --> IA[Input Analysis]
    DISPATCH --> MA[Methodology Analysis]
    DISPATCH --> IMA[Implementation Analysis]
    DISPATCH --> UA[Usage Analysis]
    
    PA --> |Section 1| COLLECT[Collect Results]
    IA --> |Section 2| COLLECT
    MA --> |Section 3| COLLECT
    IMA --> |Section 4| COLLECT
    UA --> |Section 5| COLLECT
    
    COLLECT --> SYNTH[Synthesize Document]
    SYNTH --> VALIDATE{Validate}
    
    VALIDATE -->|Pass| FORMAT[Format Markdown]
    VALIDATE -->|Fail| FIX[Fix Issues]
    FIX --> SYNTH
    
    FORMAT --> OUTPUT([Output MRMC])
    
    style START fill:#9f9
    style OUTPUT fill:#9f9
    style VALIDATE fill:#ff9
```

## 5. VS Code Integration Flow

```mermaid
graph TB
    subgraph "VS Code Environment"
        USER[User Types /mrmc]
        COPILOT[GitHub Copilot]
        EXT[MRMC Extension]
        EDITOR[Code Editor]
    end
    
    subgraph "MCP Protocol"
        MCP_CLIENT[MCP Client]
        MCP_SERVER[MCP Server]
    end
    
    subgraph "Agent System"
        AGENT[MRMC Agents]
        TOOLS[Analysis Tools]
    end
    
    subgraph "Results"
        PREVIEW[Markdown Preview]
        SAVE[Save to Branch]
        PR[Create PR]
    end
    
    USER --> COPILOT
    COPILOT --> EXT
    EXT --> MCP_CLIENT
    MCP_CLIENT -.->|"WebSocket"| MCP_SERVER
    MCP_SERVER --> AGENT
    AGENT --> TOOLS
    TOOLS -->|"Analyze repo"| AGENT
    AGENT -->|"Generate doc"| MCP_SERVER
    MCP_SERVER -.->|"Return doc"| MCP_CLIENT
    MCP_CLIENT --> EDITOR
    EDITOR --> PREVIEW
    PREVIEW -->|"User saves"| SAVE
    SAVE --> PR
    
    style USER fill:#bbf
    style PREVIEW fill:#9f9
```

## 6. CI/CD Integration Pipeline

```mermaid
flowchart LR
    subgraph "GitHub"
        PUSH[Code Push] --> TRIGGER[Trigger Workflow]
        TRIGGER --> ACTION[GitHub Action]
    end
    
    subgraph "MRMC Agent Container"
        ACTION --> PULL[Pull Repository]
        PULL --> RUN[Run Agent Bot]
        RUN --> GEN[Generate Document]
        GEN --> VAL[Validate]
    end
    
    subgraph "Results"
        VAL -->|"Success"| COMMIT[Commit to Branch]
        VAL -->|"Failure"| NOTIFY[Notify Team]
        COMMIT --> PR[Create PR]
        PR --> REVIEW[Human Review]
        REVIEW -->|"Approve"| MERGE[Merge to Main]
    end
    
    style PUSH fill:#ff9
    style MERGE fill:#9f9
```

## 7. Data Flow Through Agents

```mermaid
graph TB
    subgraph "Repository"
        CODE[Source Code]
        CONFIG[Config Files]
        DOCS[Documentation]
        TESTS[Test Files]
        GIT[Git History]
    end
    
    subgraph "Data Extraction"
        CODE --> CE[Code Extractor]
        CONFIG --> CFE[Config Extractor]
        DOCS --> DE[Docs Extractor]
        TESTS --> TE[Test Extractor]
        GIT --> GE[Git Extractor]
    end
    
    subgraph "Agent Processing"
        CE & CFE & DE & TE & GE --> DS[Data Store]
        DS --> AG[Agents]
        AG --> LLM[LLM Processing]
        LLM --> SEC[Sections]
    end
    
    subgraph "Document Assembly"
        SEC --> S1[1. Purpose]
        SEC --> S2[2. Inputs]
        SEC --> S3[3. Methodology]
        SEC --> S4[4. Implementation]
        SEC --> S5[5. Usage]
        S1 & S2 & S3 & S4 & S5 --> FINAL[Final Document]
    end
    
    style DS fill:#ff9
    style FINAL fill:#9f9
```

## 8. Error Handling & Recovery

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> Scanning
    Scanning --> Processing
    
    Processing --> AgentWork
    AgentWork --> Validation
    
    Validation --> Success: All sections valid
    Validation --> PartialFailure: Some sections invalid
    Validation --> TotalFailure: Critical error
    
    PartialFailure --> Retry: Rerun failed agents
    Retry --> AgentWork
    
    TotalFailure --> ErrorReport
    ErrorReport --> [*]
    
    Success --> GenerateDoc
    GenerateDoc --> [*]
    
    state Processing {
        [*] --> Purpose
        [*] --> Inputs
        [*] --> Methodology
        [*] --> Implementation
        [*] --> Usage
        
        Purpose --> Done
        Inputs --> Done
        Methodology --> Done
        Implementation --> Done
        Usage --> Done
    }
```

## 9. Tool Architecture

```mermaid
classDiagram
    class Tool {
        <<interface>>
        +name: str
        +description: str
        +execute(params): Result
    }
    
    class CodeSearchTool {
        +search_pattern: str
        +file_types: list
        +execute(): CodeMatches
    }
    
    class ASTAnalyzerTool {
        +language: str
        +parse_file(path): AST
        +find_classes(): Classes
        +find_functions(): Functions
    }
    
    class ConfigReaderTool {
        +read_yaml(): dict
        +read_json(): dict
        +read_toml(): dict
        +extract_params(): Params
    }
    
    class GitHistoryTool {
        +get_commits(): Commits
        +get_authors(): Authors
        +get_changes(): Changes
    }
    
    class DependencyTool {
        +scan_imports(): Imports
        +find_packages(): Packages
        +check_versions(): Versions
    }
    
    Tool <|-- CodeSearchTool
    Tool <|-- ASTAnalyzerTool
    Tool <|-- ConfigReaderTool
    Tool <|-- GitHistoryTool
    Tool <|-- DependencyTool
    
    class ToolManager {
        +tools: List[Tool]
        +register(tool)
        +execute(tool_name, params)
        +batch_execute(requests)
    }
    
    ToolManager --> Tool: manages
```

## 10. Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV_VS[VS Code + Extension]
        DEV_CLI[CLI Tool]
    end
    
    subgraph "Container Deployment"
        DOCKER[Docker Image]
        K8S[Kubernetes Pod]
    end
    
    subgraph "Cloud Services"
        LB[Load Balancer]
        API1[API Instance 1]
        API2[API Instance 2]
        API3[API Instance 3]
        CACHE[Redis Cache]
        AUDIT_S3[S3 Audit Storage]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        GITHUB[GitHub API]
    end
    
    DEV_VS & DEV_CLI --> LB
    LB --> API1 & API2 & API3
    API1 & API2 & API3 --> CACHE
    API1 & API2 & API3 --> AUDIT_S3
    API1 & API2 & API3 --> OPENAI
    API1 & API2 & API3 --> GITHUB
    
    DOCKER --> K8S
    K8S --> API1 & API2 & API3
    
    style LB fill:#ff9
    style CACHE fill:#9ff
```

## 11. Security & Audit Flow

```mermaid
flowchart TB
    subgraph "Request"
        REQ[Incoming Request] --> AUTH{Authenticate}
        AUTH -->|Valid| AUTHOR{Authorize}
        AUTH -->|Invalid| DENY[Deny Access]
        AUTHOR -->|Permitted| PROCESS
        AUTHOR -->|Forbidden| DENY
    end
    
    subgraph "Processing"
        PROCESS[Process Request] --> LOG1[Log Request]
        LOG1 --> EXEC[Execute Agents]
        EXEC --> LOG2[Log Operations]
        LOG2 --> RESULT[Generate Result]
    end
    
    subgraph "Audit Trail"
        LOG1 --> AUDIT[(Audit Database)]
        LOG2 --> AUDIT
        RESULT --> LOG3[Log Output]
        LOG3 --> AUDIT
        AUDIT --> IMMUTABLE[Immutable Storage]
    end
    
    subgraph "Compliance"
        IMMUTABLE --> REPORT[Audit Reports]
        REPORT --> REG[Regulators]
    end
    
    style AUTH fill:#ff9
    style AUDIT fill:#9f9
    style IMMUTABLE fill:#99f
```

## 12. Performance Optimization

```mermaid
graph LR
    subgraph "Caching Layer"
        REQ[Request] --> CACHE{Cache Hit?}
        CACHE -->|Yes| RETURN[Return Cached]
        CACHE -->|No| PROCESS
    end
    
    subgraph "Parallel Processing"
        PROCESS[Process] --> SPLIT[Split Work]
        SPLIT --> W1[Worker 1]
        SPLIT --> W2[Worker 2]
        SPLIT --> W3[Worker 3]
        SPLIT --> W4[Worker 4]
        SPLIT --> W5[Worker 5]
        W1 & W2 & W3 & W4 & W5 --> COMBINE[Combine]
    end
    
    subgraph "Optimization"
        COMBINE --> OPT[Optimize]
        OPT --> STORE[Store in Cache]
        STORE --> RETURN2[Return Result]
    end
    
    style CACHE fill:#9ff
    style SPLIT fill:#ff9
    style COMBINE fill:#9f9
```

These diagrams illustrate:

1. **System Overview**: How all components connect
2. **Agent Communication**: How agents collaborate
3. **Tool Usage**: How tools serve agents
4. **Generation Pipeline**: Step-by-step document creation
5. **VS Code Integration**: IDE workflow
6. **CI/CD Pipeline**: Automated generation
7. **Data Flow**: Information movement through system
8. **Error Handling**: Failure recovery mechanisms
9. **Tool Architecture**: Object-oriented design
10. **Deployment**: Production infrastructure
11. **Security & Audit**: Compliance tracking
12. **Performance**: Optimization strategies

Each diagram tells a different part of the story, making it easy to explain the system to different audiences - from developers to executives to regulators.