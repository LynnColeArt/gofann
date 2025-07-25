// Cascade training demonstration - automatically grows the network
package main

import (
	"fmt"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN Cascade Training Demonstration")
	fmt.Println("=====================================")
	fmt.Println("Cascade training starts with a minimal network and automatically")
	fmt.Println("adds hidden neurons until the desired performance is achieved.")
	fmt.Println()
	
	// Load XOR dataset
	xorPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/examples/xor.data"
	data, err := gofann.ReadTrainFromFile[float32](xorPath)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	
	// Create minimal network (no hidden layer)
	net := gofann.CreateStandard[float32](2, 1)
	fmt.Printf("Starting network: %d inputs -> %d outputs (no hidden layer)\n",
		net.GetNumInput(), net.GetNumOutput())
	fmt.Printf("Total neurons: %d\n", net.GetTotalNeurons())
	
	// Test initial performance
	initialMSE := net.TestData(data)
	fmt.Printf("Initial MSE: %.4f\n\n", initialMSE)
	
	// Configure cascade parameters
	net.SetCascadeOutputChangeFraction(0.01)      // 1% improvement required
	net.SetCascadeOutputStagnationEpochs(12)      // Stop after 12 epochs without improvement
	net.SetCascadeCandidateChangeFraction(0.01)   // 1% improvement for candidates
	net.SetCascadeCandidateStagnationEpochs(12)   // Stop candidates after 12 epochs
	net.SetCascadeMaxOutEpochs(100)               // Maximum epochs for output training
	net.SetCascadeMaxCandEpochs(100)              // Maximum epochs for candidate training
	net.SetCascadeMinOutEpochs(20)                // Minimum epochs before checking stagnation
	net.SetCascadeMinCandEpochs(20)               // Minimum epochs for candidates
	
	// Set candidate groups (try multiple random initializations)
	net.SetCascadeNumCandidateGroups(2)
	
	// Track progress
	lastNeuronCount := net.GetTotalNeurons()
	fmt.Println("Training progress:")
	fmt.Println("-----------------")
	
	net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
		currentNeurons := ann.GetTotalNeurons()
		if currentNeurons > lastNeuronCount {
			hiddenNeurons := currentNeurons - 4 // 2 input + 1 bias + 1 output
			fmt.Printf("Added hidden neuron %d - Total neurons: %d, MSE: %.6f\n", 
				hiddenNeurons, currentNeurons, mse)
			lastNeuronCount = currentNeurons
		}
		
		return true
	})
	
	// Train with cascade
	fmt.Println("\nStarting cascade training...")
	net.CascadetrainOnData(data, 10, 1, 0.001)
	
	// Show final network structure
	fmt.Printf("\nFinal network structure:\n")
	fmt.Printf("Total neurons: %d\n", net.GetTotalNeurons())
	fmt.Printf("Hidden neurons added: %d\n", neuronsAdded)
	
	// Test final performance
	finalMSE := net.TestData(data)
	fmt.Printf("\nFinal MSE: %.6f (improvement: %.1f%%)\n", 
		finalMSE, (1.0 - finalMSE/initialMSE) * 100)
	
	// Show detailed results
	fmt.Println("\nDetailed results:")
	fmt.Println("Input  | Output | Expected")
	fmt.Println("-------|--------|----------")
	
	for i := 0; i < data.GetNumData(); i++ {
		input := data.GetInput(i)
		output := net.Run(input)
		expected := data.GetOutput(i)
		
		fmt.Printf("%v | %.4f | %.1f\n", input, output[0], expected[0])
	}
	
	// Compare with fixed topology
	fmt.Println("\n\nComparison with fixed topology:")
	fmt.Println("--------------------------------")
	
	// Create standard network with hidden layer
	standardNet := gofann.CreateStandard[float32](2, 4, 1)
	standardNet.SetTrainingAlgorithm(gofann.TrainRPROP)
	
	// Train for same number of epochs
	standardMSE := standardNet.TestData(data)
	fmt.Printf("Standard network (2-4-1) initial MSE: %.4f\n", standardMSE)
	
	for i := 0; i < 500; i++ {
		standardNet.TrainEpoch(data)
	}
	
	standardMSE = standardNet.TestData(data)
	fmt.Printf("Standard network after 500 epochs: %.6f\n", standardMSE)
	
	fmt.Println("\nKey benefits of cascade training:")
	fmt.Println("- Automatically determines network size")
	fmt.Println("- Often finds smaller networks that work well")
	fmt.Println("- No need to guess the number of hidden neurons")
	fmt.Println("- Can discover non-standard architectures")
}