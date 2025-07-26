# GoFANN Project Rules

## Core Development Principles

1. **NO MOCKS OUTSIDE OF UNIT TESTS** - Never create fake/mock functionality in demos or examples. Build real working systems or clearly label architectural concepts only.

2. **Real Neural Networks Only** - When demonstrating AI functionality, use actual trained FANN networks, not hardcoded responses or keyword matching.

3. **Functional Over Theatrical** - Focus on building working functionality rather than elaborate demos with fake responses.

## Project Context

This is a Go port of the FANN (Fast Artificial Neural Network) library with extensions for:
- Reflective training (inspired by Lane Cunningham)
- Mixture of Experts (MoE) routing  
- Cross-domain knowledge fusion

## The Stacked Architecture Vision

GoFANN enables a revolutionary stacked approach to AI problem-solving, inspired by Lane Cunningham's tiny text-to-SQL model that achieved 76.91% accuracy through self-reflection.

### The Stack:

```
Layer 1: Pattern Recognition Experts (Error classification, context analysis)
    ↓
Layer 2: Strategy Selection Experts (Choose approach based on patterns)
    ↓
Layer 3: Generation Experts (Generate specific solutions/commands)
    ↓
Layer 4: Self-Doubt Experts (Validate and iterate until confident)
    ↓
Final Output: Executable solution
```

### MoE Router as Orchestrator:

The MoE router orchestrates this entire stack:
- Activates relevant experts at each layer
- Enables cross-domain fusion (e.g., git error + npm context)
- Manages confidence thresholds
- Implements reflective improvement loops

### Key Insight:

Instead of one large model trying to do everything, use many tiny specialized FANNs that:
- Excel at their specific task
- Self-reflect and improve
- Collaborate through the MoE router
- Stack together to solve complex problems

### Example Use Case:

**CLI Assistant**: Tiny specialized FANNs that understand command-line tools better than large LLMs:
- Git expert recognizes error patterns
- Strategy expert selects debug approach
- Command expert generates exact fixes
- Validator ensures safety and correctness

This architecture enables CPU-runnable AI assistants that are actually experts at their specialized domains.

## Testing Commands

- Run tests: `go test`
- Run specific test: `go test -run TestName`