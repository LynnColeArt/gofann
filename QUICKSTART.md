# GoFANN Quick Start Guide

## Installation

```bash
go get github.com/lynncoleart/gofann
```

## Manual Testing Options

### 1. Run the Test Script

```bash
# Run all tests
./test_manual.sh

# Run specific test suites
./test_manual.sh basic       # Basic XOR/AND tests
./test_manual.sh reflective  # Reflective training tests
./test_manual.sh concurrent  # Concurrent training tests
./test_manual.sh cascade     # Cascade network tests
./test_manual.sh benchmark   # Performance benchmarks
./test_manual.sh example     # Run example program
./test_manual.sh cli         # CLI assistant demo
./test_manual.sh interactive # Interactive REPL
```

### 2. Run Individual Tests

```bash
# Test basic functionality
go test -v -run TestXORTraining

# Test reflective training
go test -v -run TestReflectiveTrainer

# Test concurrent training
go test -v -run TestConcurrentExpertTraining

# Run benchmarks
go test -bench=BenchmarkXORTraining -benchtime=10s
```

### 3. Run the Examples

```bash
# Basic manual test
cd examples
go run manual_test.go

# CLI Assistant
cd cli_assistant
go run main.go
```

### 4. Interactive Testing

The interactive mode lets you experiment with networks:

```bash
./test_manual.sh interactive

# Commands:
# > xor     - Create XOR network
# > and     - Create AND network  
# > train   - Train the network
# > test    - Test with custom inputs
# > quit    - Exit
```

### 5. Quick Code Example

```go
package main

import (
    "fmt"
    "github.com/lynncoleart/gofann"
)

func main() {
    // Create a 2-4-1 network
    net := gofann.CreateStandard[float32](2, 4, 1)
    net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
    net.SetActivationFunctionOutput(gofann.Sigmoid)
    net.RandomizeWeights(-1, 1)
    
    // Create XOR training data
    inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    outputs := [][]float32{{0}, {1}, {1}, {0}}
    data := gofann.CreateTrainDataArray(inputs, outputs)
    
    // Train
    net.TrainOnData(data, 1000, 100, 0.01)
    
    // Test
    for i, input := range inputs {
        result := net.Run(input)
        fmt.Printf("XOR(%v) = %.3f (expected %.0f)\n", 
            input, result[0], outputs[i][0])
    }
}
```

### 6. Monitor Training Progress

```go
// Create a training monitor
monitor := gofann.NewTrainingMonitor[float32]()
monitor.Start(1000) // 1000 epochs

// Set callback to update monitor
net.SetCallback(func(ann *gofann.Fann[float32], epoch int, mse float32) bool {
    monitor.Update(epoch, mse, accuracy, ann.GetLearningRate())
    return true
})

// Train with monitoring
net.TrainOnData(data, 1000, 10, 0.01)
monitor.Complete()
```

### 7. Test Reflective Training

```go
// Create network and reflective trainer
net := gofann.CreateStandard[float32](10, 20, 4)
trainer := gofann.NewReflectiveTrainer(net)

// Monitor weaknesses
trainer.OnWeaknessDetected = func(weaknesses []gofann.Weakness[float32]) {
    fmt.Printf("Found %d weaknesses\n", len(weaknesses))
    for _, w := range weaknesses {
        fmt.Printf("  - %s (%.1f%% confusion)\n", 
            w.Pattern, w.ConfusionRate*100)
    }
}

// Train with reflection
metrics := trainer.TrainWithReflection(data)
```

## What to Look For

When testing manually, check for:

1. **MSE Decreasing**: Should go down during training
2. **Output Range**: Should be 0-1 for Sigmoid output
3. **Convergence**: XOR should reach < 0.01 MSE
4. **Performance**: Training should be fast (< 1 second)
5. **Memory**: Should use minimal memory

## Troubleshooting

If networks aren't training:
- Check activation functions are set
- Verify weights are randomized
- Ensure learning rate is reasonable (0.1-0.7)
- Try different training algorithms (RPROP usually works best)

## Next Steps

1. Try the reflective training system
2. Test concurrent multi-expert training
3. Build your own CLI error classifier
4. Benchmark against your use case