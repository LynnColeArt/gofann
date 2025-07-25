# Cascade Training Redesign

## Key Differences from Current Implementation

Based on analysis of the original FANN cascade training:

### 1. Candidate Structure
**Current**: Candidates only have input connections
**Correct**: Candidates must have BOTH input connections AND output connections

### 2. Training Objective
**Current**: Maximize correlation between candidate activation and output error
**Correct**: Minimize squared difference between (candidate_activation * output_weight) and output_error

### 3. Weight Installation
**Current**: Random output weights when installing candidate
**Correct**: Use the trained output weights from candidate training

## Implementation Plan

### Phase 1: Update Candidate Structure
- Modify candidate neurons to include output connections
- Allocate space for candidate output weights
- Initialize candidate output weights properly

### Phase 2: Fix Candidate Training
- Implement the proper objective function: minimize sum((activation * weight - error)²)
- Update gradients for both input and output weights
- Use the cascade_weight_multiplier parameter

### Phase 3: Fix Candidate Installation  
- Preserve trained output weights when installing
- Properly update network structure
- Maintain training state (RPROP steps, etc.)

## Algorithm Overview

```
for each epoch:
    // Forward pass
    run network with input
    calculate output errors
    
    // For each candidate
    for each candidate:
        // Forward pass through candidate
        sum = 0
        for each input connection:
            sum += weight * neuron_value
        activation = activate(sum)
        
        // Calculate score and gradients
        score = MSE  // Start with current MSE
        for each output:
            diff = (activation * output_weight) - output_error
            score -= diff²  // Higher score is better
            
            // Gradient for output weight
            output_slope += -2 * diff * activation
            
            // Gradient for input weights
            error_value += diff * output_weight
        
        error_value *= activation_derivative
        for each input connection:
            input_slope += error_value * input_neuron_value
```

## Key Parameters
- `cascade_weight_multiplier`: Initial magnitude for output weights
- `cascade_candidate_limit`: Max correlation/MSE ratio
- Training uses RPROP, Quickprop, or Sarprop (not Batch/Incremental)