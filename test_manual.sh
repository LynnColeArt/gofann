#!/bin/bash

echo "🧠 GoFANN Manual Testing Suite"
echo "=============================="
echo

# Different test options
case "${1:-all}" in
    "basic")
        echo "Running basic tests..."
        go test -v -run "TestXOR|TestAND|TestDebugSimple" -count=1
        ;;
    
    "reflective")
        echo "Running reflective training tests..."
        go test -v -run "TestReflective" -count=1
        ;;
    
    "concurrent")
        echo "Running concurrent training tests..."
        go test -v -run "TestConcurrent" -count=1
        ;;
    
    "cascade")
        echo "Running cascade tests..."
        go test -v -run "TestCascade" -count=1
        ;;
    
    "benchmark")
        echo "Running benchmarks..."
        go test -bench=. -benchtime=10s -run=XXX
        ;;
    
    "example")
        echo "Running manual example..."
        cd examples
        go run manual_test.go
        cd ..
        ;;
    
    "cli")
        echo "Running CLI assistant example..."
        cd examples/cli_assistant
        go run main.go
        cd ../..
        ;;
    
    "interactive")
        echo "Starting interactive test session..."
        echo "Creating test program..."
        cat > /tmp/gofann_test.go << 'EOF'
package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
    "github.com/lynncoleart/gofann"
)

func main() {
    reader := bufio.NewReader(os.Stdin)
    
    fmt.Println("🧠 GoFANN Interactive Tester")
    fmt.Println("Commands: xor, and, train, test, quit")
    fmt.Println()
    
    var net *gofann.Fann[float32]
    
    for {
        fmt.Print("> ")
        cmd, _ := reader.ReadString('\n')
        cmd = strings.TrimSpace(cmd)
        
        switch cmd {
        case "xor":
            net = createXORNetwork()
            fmt.Println("Created XOR network")
            
        case "and":
            net = createANDNetwork()
            fmt.Println("Created AND network")
            
        case "train":
            if net == nil {
                fmt.Println("Create a network first!")
                continue
            }
            trainNetwork(net)
            
        case "test":
            if net == nil {
                fmt.Println("Create a network first!")
                continue
            }
            testNetwork(net, reader)
            
        case "quit", "exit":
            fmt.Println("Goodbye!")
            return
            
        default:
            fmt.Println("Unknown command:", cmd)
        }
    }
}

func createXORNetwork() *gofann.Fann[float32] {
    net := gofann.CreateStandard[float32](2, 4, 1)
    net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
    net.SetActivationFunctionOutput(gofann.Sigmoid)
    net.RandomizeWeights(-1, 1)
    net.SetLearningRate(0.7)
    return net
}

func createANDNetwork() *gofann.Fann[float32] {
    net := gofann.CreateStandard[float32](2, 3, 1)
    net.SetActivationFunctionHidden(gofann.SigmoidSymmetric)
    net.SetActivationFunctionOutput(gofann.Sigmoid)
    net.RandomizeWeights(-1, 1)
    net.SetLearningRate(0.7)
    return net
}

func trainNetwork(net *gofann.Fann[float32]) {
    fmt.Println("Training on XOR data...")
    
    inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    outputs := [][]float32{{0}, {1}, {1}, {0}}
    data := gofann.CreateTrainDataArray(inputs, outputs)
    
    net.TrainOnData(data, 1000, 100, 0.01)
    
    fmt.Println("Training complete! Final MSE:", net.GetMSE())
}

func testNetwork(net *gofann.Fann[float32], reader *bufio.Reader) {
    fmt.Println("Enter two inputs (0 or 1), e.g.: 1 0")
    fmt.Print("Input: ")
    
    input, _ := reader.ReadString('\n')
    parts := strings.Fields(input)
    
    if len(parts) != 2 {
        fmt.Println("Please enter exactly 2 numbers")
        return
    }
    
    var inputs []float32
    for _, p := range parts {
        val, err := strconv.ParseFloat(p, 32)
        if err != nil {
            fmt.Println("Invalid number:", p)
            return
        }
        inputs = append(inputs, float32(val))
    }
    
    output := net.Run(inputs)
    fmt.Printf("Network output: %.3f\n", output[0])
}
EOF
        go run /tmp/gofann_test.go
        ;;
    
    "all"|*)
        echo "Running all tests..."
        echo
        echo "1. Basic Tests"
        echo "--------------"
        go test -v -run "TestXOR|TestAND" -count=1
        echo
        echo "2. Reflective Training"
        echo "---------------------"
        go test -v -run "TestReflective" -count=1 | head -20
        echo
        echo "3. Concurrent Training"
        echo "---------------------"
        go test -v -run "TestConcurrentExpert" -count=1
        echo
        echo "4. Running Example"
        echo "------------------"
        cd examples
        go run manual_test.go
        cd ..
        ;;
esac

echo
echo "✅ Testing complete!"
echo
echo "Usage: $0 [basic|reflective|concurrent|cascade|benchmark|example|cli|interactive|all]"