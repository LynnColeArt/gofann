# Cascade Training Implementation

Cascade training has been successfully implemented in GoFANN. This feature allows networks to automatically grow by adding hidden neurons until the desired performance is achieved.

## Key Features Implemented

1. **Core Functions**:
   - `CascadetrainOnData()` - Main cascade training function
   - `CreateCascade()` - Helper to create minimal networks for cascade training
   - Support for both creating new hidden layers and adding to existing ones

2. **Cascade Parameters** (with getters/setters):
   - Output change fraction
   - Output stagnation epochs
   - Candidate change fraction
   - Candidate stagnation epochs
   - Candidate limit
   - Max/Min epochs for output and candidate training
   - Activation functions for candidates
   - Activation steepnesses for candidates
   - Number of candidate groups

3. **Algorithm Features**:
   - Trains output connections between adding neurons
   - Trains multiple candidate neurons with different activation functions
   - Selects best candidate based on correlation score
   - Preserves trained weights when installing candidates
   - Supports shortcut connections from all previous layers

## Usage Example

```go
// Create minimal network for cascade training
net := gofann.CreateCascade[float32](2, 1)

// Or manually configure
net := gofann.CreateStandard[float32](2, 1)
net.SetCascadeMaxOutEpochs(100)
net.SetCascadeOutputStagnationEpochs(12)
// ... set other parameters

// Train with cascade
net.CascadetrainOnData(trainData, maxNeurons, neuronsPerCascade, desiredError)
```

## Implementation Notes

- The algorithm starts with a minimal network (no hidden layer)
- Candidate neurons are trained to correlate with output errors
- Best candidate is installed with its trained weights
- Network topology is automatically adjusted as neurons are added
- Supports both float32 and float64 through generics

## Testing

- Basic cascade test validates neuron addition
- Parameter tests verify all getters/setters
- Larger dataset test shows cascade behavior on real data
- Demo program illustrates cascade training on XOR problem

## Known Limitations

- Cascade training may not always find optimal solutions for simple problems like XOR
- The correlation-based candidate selection may need tuning for specific problems
- Performance depends heavily on parameter settings