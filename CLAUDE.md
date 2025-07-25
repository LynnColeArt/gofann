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

## Testing Commands

- Run tests: `go test`
- Run specific test: `go test -run TestName`