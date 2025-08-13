# MRMC Governance Bot: Architecture Diagrams

## Overview

This document presents the architectural diagrams for the redesigned MRMC Bot as a comprehensive model governance platform that handles steps 3-5 of the governance workflow, with human approval as the gateway between documentation and advanced assessments.

---

## 1. Complete Governance Workflow (Corrected)

```mermaid
graph TB
    subgraph "Manual Pre-Work"
        REG[1. Register Model in Inventory]
        RISK[2. Inherent Risk Rating Assessment]
    end
    
    subgraph "MRMC Governance Bot Platform"
        DOC[3. Generate MRMC Documentation]
        GATE{Human Review & Approval}
        IMMAT[4. Immateriality Assessment]
        KPI[5. Performance Concepts & KPIs]
    end
    
    subgraph "Outputs"
        PACKAGE[Complete Governance Package]
    end
    
    REG -->|Model ID, Business Context| RISK
    RISK -->|Risk Level, Compliance Needs| DOC
    DOC -->|Draft Documentation| GATE
    GATE -->|Approved| IMMAT
    GATE -->|Rejected| DOC
    IMMAT -->|Materiality Score| KPI
    KPI -->|Performance Framework| PACKAGE
    
    style REG fill:#e1f5fe
    style RISK fill:#e1f5fe
    style DOC fill:#f3e5f5
    style GATE fill:#fff3e0
    style IMMAT fill:#e8f5e8
    style KPI fill:#e8f5e8
    style PACKAGE fill:#f1f8e9
```

**Key Changes:**
- Bot now owns steps 3-5 as a unified platform
- Human approval is the gateway that unlocks phases 2 & 3
- Clear progression: Manual → Documentation → Approval → Assessment → Performance

---

## 2. MRMC Bot Three-Phase Architecture

```mermaid
graph TB
    subgraph "Inputs"
        MI[Model Inventory Data]
        RA[Risk Assessment Data]
        CODE[Repository Code]
    end
    
    subgraph "MRMC Governance Bot"
        subgraph "Phase 1: Documentation Generation"
            PA[Purpose Agent]
            IA[Inputs Agent]
            MA[Methodology Agent]
            IMA[Implementation Agent]
            UA[Usage Agent]
        end
        
        APPROVAL{Human Approval Gateway}
        
        subgraph "Phase 2: Immateriality Assessment"
            MATA[Materiality Agent]
            COMP[Complexity Agent]
            IMP[Impact Agent]
        end
        
        subgraph "Phase 3: Performance Framework"
            KPIA[KPI Agent]
            METR[Metrics Agent]
            THR[Threshold Agent]
            MON[Monitoring Agent]
        end
        
        ORC[Orchestrator]
    end
    
    subgraph "Outputs"
        DOCS[MRMC Documentation]
        SCORE[Materiality Score]
        PERF[Performance Framework]
    end
    
    MI & RA & CODE --> PA & IA & MA & IMA & UA
    PA & IA & MA & IMA & UA --> ORC
    ORC --> DOCS
    DOCS --> APPROVAL
    
    APPROVAL -->|Approved| MATA & COMP & IMP
    APPROVAL -->|Rejected| PA & IA & MA & IMA & UA
    
    MATA & COMP & IMP --> ORC
    ORC --> SCORE
    SCORE --> KPIA & METR & THR & MON
    KPIA & METR & THR & MON --> ORC
    ORC --> PERF
    
    style PA fill:#f3e5f5
    style IA fill:#f3e5f5
    style MA fill:#f3e5f5
    style IMA fill:#f3e5f5
    style UA fill:#f3e5f5
    style APPROVAL fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style MATA fill:#e8f5e8
    style COMP fill:#e8f5e8
    style IMP fill:#e8f5e8
    style KPIA fill:#e3f2fd
    style METR fill:#e3f2fd
    style THR fill:#e3f2fd
    style MON fill:#e3f2fd
    style ORC fill:#fce4ec,stroke:#e91e63,stroke-width:3px
```

---

## 3. Human Approval Gateway Detail

```mermaid
stateDiagram-v2
    [*] --> DocumentationGenerated
    DocumentationGenerated --> UnderReview: Submit for Review
    
    state UnderReview {
        [*] --> QualityCheck
        QualityCheck --> TechnicalReview: Pass
        TechnicalReview --> BusinessValidation: Pass
        BusinessValidation --> ComplianceCheck: Pass
        ComplianceCheck --> [*]: Pass
    }
    
    UnderReview --> NeedsRevision: Fail any check
    UnderReview --> Approved: All checks pass
    
    state NeedsRevision {
        [*] --> EditingMode
        EditingMode --> RegenerateSection: Major changes
        EditingMode --> MinorEdits: Small fixes
        RegenerateSection --> [*]
        MinorEdits --> [*]
    }
    
    NeedsRevision --> UnderReview: Resubmit
    Approved --> ImmaterialityPhase: Unlock Phase 2
    ImmaterialityPhase --> PerformancePhase: Continue
    PerformancePhase --> Complete
    Complete --> [*]
    
    note right of UnderReview
        Quality Checks:
        - Technical accuracy
        - Business context
        - Regulatory compliance
        - Completeness
    end note
    
    note right of Approved
        Gateway Effect:
        Approval unlocks
        advanced bot phases
    end note
```

---

## 4. Phase 2: Immateriality Assessment Agents

```mermaid
graph LR
    subgraph "Immateriality Assessment Input"
        DOCS[Approved MRMC Documentation]
        RISK[Risk Assessment Data]
        USAGE[Usage Patterns]
    end
    
    subgraph "Assessment Agents"
        MATA[Materiality Agent]
        COMP[Complexity Agent]
        IMP[Impact Agent]
    end
    
    subgraph "Assessment Dimensions"
        FIN[Financial Impact]
        REG[Regulatory Exposure]
        OP[Operational Criticality]
        FREQ[Usage Frequency]
        SCOPE[Decision Scope]
        COMPLEX[Model Complexity]
    end
    
    subgraph "Materiality Calculation"
        CALC[Weighted Scoring Engine]
        CLASS[Materiality Classification]
    end
    
    DOCS --> MATA
    DOCS --> COMP
    DOCS --> IMP
    RISK --> MATA & IMP
    USAGE --> MATA & IMP
    
    MATA --> FIN & REG & OP
    COMP --> COMPLEX
    IMP --> FREQ & SCOPE
    
    FIN & REG & OP & FREQ & SCOPE & COMPLEX --> CALC
    CALC --> CLASS
    
    CLASS --> IMMATERIAL[Immaterial<br/>Score: 0-30]
    CLASS --> LOW[Low Material<br/>Score: 31-50]
    CLASS --> MODERATE[Moderate<br/>Score: 51-70]
    CLASS --> HIGH[High Material<br/>Score: 71-85]
    CLASS --> CRITICAL[Critical<br/>Score: 86-100]
    
    style MATA fill:#e8f5e8
    style COMP fill:#e8f5e8
    style IMP fill:#e8f5e8
    style CALC fill:#fff3e0
    style IMMATERIAL fill:#c8e6c9
    style LOW fill:#dcedc8
    style MODERATE fill:#fff9c4
    style HIGH fill:#ffcc02
    style CRITICAL fill:#ff8a65
```

---

## 5. Phase 3: Performance Framework Agents

```mermaid
graph TB
    subgraph "Performance Framework Input"
        DOCS2[MRMC Documentation]
        MAT[Materiality Score]
        RISKDATA[Risk Level]
    end
    
    subgraph "Performance Agents"
        KPIA[KPI Agent]
        METR[Metrics Agent]
        THR[Threshold Agent]
        MON[Monitoring Agent]
    end
    
    subgraph "KPI Categories"
        PERF[Performance Metrics]
        STAB[Stability Indicators]
        FAIR[Fairness Measures]
        BUS[Business Outcomes]
    end
    
    subgraph "Framework Components"
        DEF[Metric Definitions]
        CALC2[Calculation Methods]
        THRESH[Thresholds & Limits]
        ALERT[Alerting Rules]
        DASH[Dashboard Config]
    end
    
    DOCS2 --> KPIA
    MAT --> KPIA & THR & MON
    RISKDATA --> KPIA & THR
    
    KPIA --> PERF & STAB & FAIR & BUS
    METR --> DEF & CALC2
    THR --> THRESH
    MON --> ALERT & DASH
    
    PERF --> DEF
    STAB --> DEF
    FAIR --> DEF
    BUS --> DEF
    
    DEF & CALC2 & THRESH & ALERT & DASH --> FRAMEWORK[Complete Performance Framework]
    
    style KPIA fill:#e3f2fd
    style METR fill:#e3f2fd
    style THR fill:#e3f2fd
    style MON fill:#e3f2fd
    style FRAMEWORK fill:#f1f8e9
```

---

## 6. Complete Bot Platform Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Bot as MRMC Governance Bot
    participant P1 as Phase 1 Agents
    participant Review as Human Reviewer
    participant P2 as Phase 2 Agents
    participant P3 as Phase 3 Agents
    participant Output as Governance Package
    
    User->>Bot: Submit model (Steps 1&2 complete)
    Bot->>P1: Generate documentation
    P1->>P1: Analyze code, risk, context
    P1->>Bot: MRMC documentation draft
    Bot->>Review: Present for approval
    
    alt Documentation Approved
        Review->>Bot: Approve documentation
        Bot->>P2: Assess immateriality
        P2->>P2: Calculate materiality score
        P2->>Bot: Materiality classification
        
        Bot->>P3: Generate performance framework
        P3->>P3: Define KPIs, metrics, thresholds
        P3->>Bot: Performance framework
        
        Bot->>Output: Complete governance package
        Output->>User: Ready for deployment
        
    else Documentation Rejected
        Review->>Bot: Request changes
        Bot->>P1: Regenerate with feedback
        P1->>Bot: Updated documentation
        Bot->>Review: Resubmit for approval
    end
    
    Note over Bot: Three-phase operation:<br/>Doc → Approval → Immateriality → Performance
```

---

## 7. Future State Architecture Vision

```mermaid
graph TB
    subgraph "Current Implementation (Phase 1)"
        CURRENT[Documentation Generation<br/>5 Specialized Agents<br/>LLM Reasoning<br/>✅ Available Now]
    end
    
    subgraph "Planned Enhancement 1 (Phase 2)"
        PHASE2[Immateriality Assessment<br/>3 Assessment Agents<br/>Materiality Scoring<br/>🚧 6-8 weeks]
    end
    
    subgraph "Planned Enhancement 2 (Phase 3)"
        PHASE3[Performance Framework<br/>4 Performance Agents<br/>KPI Generation<br/>📋 10-12 weeks]
    end
    
    subgraph "Integration Layer"
        GATEWAY[Human Approval Gateway<br/>Review Dashboard<br/>Edit Capabilities<br/>🚧 4-6 weeks]
    end
    
    subgraph "Platform Architecture"
        ORCH[Enhanced Orchestrator<br/>Multi-phase coordination<br/>State management<br/>🚧 Continuous]
    end
    
    CURRENT --> GATEWAY
    GATEWAY --> PHASE2
    PHASE2 --> PHASE3
    ORCH -.-> CURRENT & GATEWAY & PHASE2 & PHASE3
    
    style CURRENT fill:#e8f5e8,stroke:#4caf50,stroke-width:3px
    style PHASE2 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style PHASE3 fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style GATEWAY fill:#fce4ec,stroke:#e91e63,stroke-width:2px
    style ORCH fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
```

---

## 8. Agent Interaction Patterns

```mermaid
graph TB
    subgraph "Phase 1: Documentation Agents"
        PA1[Purpose Agent]
        IA1[Inputs Agent] 
        MA1[Methodology Agent]
        IMA1[Implementation Agent]
        UA1[Usage Agent]
    end
    
    subgraph "Phase 2: Assessment Agents"
        MATA1[Materiality Agent]
        COMP1[Complexity Agent]
        IMP1[Impact Agent]
    end
    
    subgraph "Phase 3: Performance Agents"
        KPIA1[KPI Agent]
        METR1[Metrics Agent]
        THR1[Threshold Agent]
        MON1[Monitoring Agent]
    end
    
    subgraph "Shared Resources"
        LLM[LLM Service]
        TOOLS[Analysis Tools]
        CONTEXT[Shared Context Store]
    end
    
    PA1 & IA1 & MA1 & IMA1 & UA1 -.-> LLM
    PA1 & IA1 & MA1 & IMA1 & UA1 -.-> TOOLS
    
    MATA1 & COMP1 & IMP1 -.-> LLM
    MATA1 & COMP1 & IMP1 -.-> CONTEXT
    
    KPIA1 & METR1 & THR1 & MON1 -.-> LLM
    KPIA1 & METR1 & THR1 & MON1 -.-> CONTEXT
    
    PA1 --> CONTEXT
    MA1 --> CONTEXT
    CONTEXT --> MATA1
    CONTEXT --> KPIA1
    
    style LLM fill:#fff3e0
    style TOOLS fill:#f3e5f5
    style CONTEXT fill:#e8f5e8
```

---

## Key Architecture Decisions

### 1. **Bot as Comprehensive Platform**
- Single bot handles all automated governance steps (3-5)
- Unified orchestrator manages multi-phase workflow
- Shared context between phases eliminates redundant analysis

### 2. **Human Approval as Gateway**
- Documentation must be approved before advanced phases
- Approval unlocks immateriality and performance assessment
- Maintains human oversight while maximizing automation

### 3. **Progressive Enhancement**
- Phase 1 (Documentation): Available now
- Phase 2 (Immateriality): 6-8 weeks development
- Phase 3 (Performance): 10-12 weeks development
- Gateway interface: 4-6 weeks development

### 4. **Agent Specialization**
- Documentation agents: Code analysis and compliance writing
- Assessment agents: Materiality calculation and classification
- Performance agents: KPI definition and monitoring framework
- Shared LLM reasoning across all phases

### 5. **Data Flow Optimization**
- Phase 1 output becomes Phase 2 input
- Phase 2 classification drives Phase 3 configuration
- Shared context store eliminates redundant processing

What do you think of this architecture? Would you like me to modify any of the diagrams or add additional views?