# GoFANN - Fast Artificial Neural Network Library for Go

[![Go Version](https://img.shields.io/badge/Go-%3E%3D%201.21-blue.svg)](https://golang.org)
[![License](https://img.shields.io/badge/License-LGPL--2.1-green.svg)](LICENSE)
[![GoDoc](https://godoc.org/github.com/LynnColeArt/gofann?status.svg)](https://godoc.org/github.com/LynnColeArt/gofann)

GoFANN is a complete Go port of the [Fast Artificial Neural Network (FANN) Library](http://leenissen.dk/fann/wp/), providing a simple, fast, and effective way to create and train neural networks in Go. Built with modern Go practices including generics for type safety and performance.

## 🚀 Features

### Core Neural Networks
- **Multiple Architectures**: Standard fully-connected, sparse, and shortcut networks
- **Cascade-Correlation**: Automatic network growth during training
- **20+ Activation Functions**: Sigmoid, Tanh, ReLU, Gaussian, Elliot, and more
- **Type Safety**: Go generics support for `float32` and `float64`

### Training Algorithms
- **Incremental Training**: Online learning with single samples
- **Batch Training**: Full dataset gradient descent
- **RPROP**: Resilient backpropagation with adaptive learning rates
- **Quickprop**: Fast quasi-Newton optimization
- **SARPROP**: Simulated annealing resilient propagation

### Advanced Features
- **File I/O**: Full compatibility with original FANN file formats
- **Network Scaling**: Input/output normalization and scaling
- **Weight Management**: Individual connection manipulation
- **Debug Utilities**: Connection inspection and parameter visualization
- **Comprehensive Testing**: 500+ tests ensuring reliability

### 🧠 Revolutionary AI Capabilities (New!)
- **Reflective Training**: Self-improving neural networks inspired by Lane Cunningham
- **Mixture of Experts (MoE)**: Intelligent routing between specialized networks
- **Cross-Domain Fusion**: Combine expertise from multiple domains
- **Metacognitive Learning**: Networks that understand their own weaknesses
- **Stacked Architecture**: Tiny specialized FANNs working together

## 📦 Installation

```bash
go get github.com/LynnColeArt/gofann
```

## 🎯 Quick Start

### Creating and Training a Network

```go
package main

import (
    "fmt"
    "github.com/LynnColeArt/gofann"
)

func main() {
    // Create a 2-3-1 network (2 inputs, 3 hidden, 1 output)
    net := gofann.CreateStandard[float32](2, 3, 1)
    defer net.Destroy()
    
    // Set training parameters
    net.SetLearningRate(0.7)
    net.SetTrainingAlgorithm(gofann.TrainRPROP)
    
    // Create XOR training data
    inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    outputs := [][]float32{{0}, {1}, {1}, {0}}
    trainData := gofann.CreateTrainDataArray(inputs, outputs)
    
    // Train the network
    net.TrainOnData(trainData, 1000, 100, 0.001)
    
    // Test the network
    for i, input := range inputs {
        result := net.Run(input)
        fmt.Printf("XOR(%v) = %.3f (expected %.0f)\n", 
            input, result[0], outputs[i][0])
    }
}
```

### Cascade-Correlation Training

```go
// Create a minimal network for cascade training
net := gofann.CreateCascade[float32](2, 1)

// Configure cascade parameters
net.SetCascadeOutputChangeFraction(0.01)
net.SetCascadeMaxOutEpochs(150)

// Let cascade training automatically add hidden neurons
net.CascadetrainOnData(trainData, 30, 5, 0.001)

fmt.Printf("Network grew to %d neurons\n", net.GetTotalNeurons())
```

### File I/O

```go
// Save network
net.Save("my_network.net")

// Load network
loadedNet, err := gofann.CreateFromFile[float32]("my_network.net")
if err != nil {
    log.Fatal(err)
}

// Train directly from file
err = net.TrainOnFile("training_data.train", 1000, 100, 0.001)
```

## 🏗️ Network Architectures

### Standard Networks
```go
// Fully connected layers
net := gofann.CreateStandard[float32](2, 5, 3, 1)
```

### Sparse Networks
```go
// 50% connection rate
net := gofann.CreateSparse[float32](0.5, 2, 10, 1)
```

### Shortcut Networks
```go
// Each layer connects to all following layers
net := gofann.CreateShortcut[float32](2, 5, 1)
```

## ⚡ Performance

GoFANN is designed for both ease of use and performance:

- **Memory Efficient**: Contiguous weight storage, minimal allocations
- **CPU Optimized**: Vectorized operations where possible
- **Type Safe**: Go generics eliminate boxing/unboxing overhead
- **Concurrent Ready**: Thread-safe operations for parallel processing

### Benchmarks vs Original FANN
*(Benchmarks coming soon - we need to implement these!)*

## 🔧 Advanced Usage

### Custom Activation Functions
```go
// Set different activation functions per layer
net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
net.SetActivationFunctionOutput(gofann.Linear)

// Individual neuron control
net.SetActivationFunction(gofann.Gaussian, 1, 0) // layer 1, neuron 0
```

### Network Scaling
```go
// Automatic scaling from training data
net.SetScalingFromData(trainData)

// Manual scaling
inputMin := []float32{-1, -1}
inputMax := []float32{1, 1}
net.SetInputScaling(inputMin, inputMax)

// Use scaled training
output := net.RunScaled(input)
```

### Weight Manipulation
```go
// Get all weights
weights := net.GetWeights()

// Set individual connection weight
net.SetWeight(fromNeuron, toNeuron, 0.5)

// Inspect network structure
net.PrintConnections()
net.PrintParameters()
```

## 🎓 Reflective Training System

Inspired by Lane Cunningham's breakthrough work showing that tiny models that understand themselves can outperform large models, GoFANN introduces metacognitive training:

```go
// Create a network with reflective training
net := gofann.CreateStandard[float32](10, 20, 4)
trainer := gofann.NewReflectiveTrainer(net)

// Train with self-reflection
metrics := trainer.TrainWithReflection(trainData)

// The network automatically:
// 1. Analyzes its confusion matrix
// 2. Identifies weakness patterns  
// 3. Generates targeted training data
// 4. Adapts learning rates based on progress
// 5. Evolves through metacognitive loops
```

## 🌟 Mixture of Experts (MoE) System

Build systems where multiple specialized networks collaborate:

```go
// Create domain experts
gitExpert := gofann.NewReflectiveExpert[float32](
    "GitMaster", "git", []int{25, 20, 10, 4})
npmExpert := gofann.NewReflectiveExpert[float32](
    "NPMWizard", "npm", []int{25, 20, 10, 4})

// Create MoE router
router := gofann.NewMoERouter([]Expert{gitExpert, npmExpert})

// Router intelligently selects and combines experts
output, decision := router.Route(input)
fmt.Printf("Experts used: %v (confidence: %.2f%%)\n", 
    decision.SelectedExperts(), decision.Confidence()*100)
```

### Real-World Example: CLI Error Assistant

```go
// Train experts on different error domains
trainer := NewCLIExpertTrainer()
trainer.TrainAllExperts("./cli-training-corpus")

// Diagnose errors with expert fusion
trainer.Diagnose("CONFLICT (content): Merge conflict in app.js")
// Output:
// 🎯 Expert Analysis:
//    Active Experts: [GitMaster, ConflictResolver]
//    Error Type: Conflict (95% confidence)
// 💡 Recommended Actions:
//    1. View conflicts with 'git status'
//    2. Edit files to resolve conflicts
//    3. Mark resolved with 'git add'
```

## 📊 Training Algorithms Comparison

| Algorithm | Speed | Memory | Convergence | Best For |
|-----------|--------|---------|-------------|----------|
| Incremental | Fast | Low | Good | Online learning |
| Batch | Medium | Medium | Stable | Small datasets |
| RPROP | Fast | Medium | Excellent | Most problems |
| Quickprop | Very Fast | Medium | Good | Well-behaved functions |
| SARPROP | Medium | High | Robust | Noisy data |

## 🧪 Testing

Run the comprehensive test suite:

```bash
go test ./...
go test -run TestCascade  # Cascade-specific tests  
go test -bench=.          # Benchmarks
```

## 📈 Roadmap

### Completed Features ✅
- **Reflective Training**: Self-improving networks with metacognitive loops
- **Mixture of Experts**: Multi-domain expert coordination
- **Fixed-Point Arithmetic**: Embedded deployment support
- **Shortcut Networks**: Skip connections for complex architectures
- **Cascade Improvements**: RPROP/Quickprop for candidate training

### Planned Improvements
- **Concurrent Training**: Parallel expert and batch processing
- **Modern Optimizers**: Adam, AdaGrad, RMSprop
- **Regularization**: Dropout, L1/L2, batch normalization  
- **Advanced Architectures**: CNN layers, RNN/LSTM support
- **GPU Acceleration**: CUDA/OpenCL backends
- **Streaming I/O**: Large dataset handling
- **Visualization**: Training progress and network topology
- **Self-Doubt Networks**: Validation layer for generated outputs

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

GoFANN is licensed under the GNU Lesser General Public License v2.1 - see [LICENSE](LICENSE) file for details.

This ensures compatibility with the original FANN library while allowing commercial use.

## 🙏 Acknowledgments

- Original [FANN Library](http://leenissen.dk/fann/wp/) by Steffen Nissen
- Cascade-Correlation algorithm by Scott Fahlman
- [Lane Cunningham](https://www.linkedin.com/in/lanecunningham/) for the revolutionary reflective training concept
- All contributors to the neural network research community

## 📚 References

- [FANN Documentation](http://leenissen.dk/fann/html/)
- [Cascade-Correlation: A new architecture and supervised learning algorithm](http://www.cs.cmu.edu/afs/cs/project/connect/bench/contrib/CMU/papers/fahlman90.pdf)
- [RPROP: A direct adaptive method for faster backpropagation learning](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417)

---

**Built with ❤️ and lots of ☕ by [Lynn Cole](https://github.com/LynnColeArt)**