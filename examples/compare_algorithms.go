// Compare different training algorithms
package main

import (
	"fmt"
	"time"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN Training Algorithm Comparison")
	fmt.Println("====================================")
	
	// Load XOR dataset
	xorPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/examples/xor.data"
	data, err := gofann.ReadTrainFromFile[float32](xorPath)
	if err != nil {
		fmt.Printf("Error loading data: %v\n", err)
		return
	}
	
	// Scale to [0,1] for sigmoid activation
	data.ScaleInput(0, 1)
	data.ScaleOutput(0, 1)
	
	algorithms := []struct {
		name string
		algo gofann.TrainAlgorithm
		setup func(net *gofann.Fann[float32])
	}{
		{
			name: "Incremental",
			algo: gofann.TrainIncremental,
			setup: func(net *gofann.Fann[float32]) {
				net.SetLearningRate(0.7)
			},
		},
		{
			name: "Batch",
			algo: gofann.TrainBatch,
			setup: func(net *gofann.Fann[float32]) {
				net.SetLearningRate(0.7)
				net.SetLearningMomentum(0.1)
			},
		},
		{
			name: "RPROP",
			algo: gofann.TrainRPROP,
			setup: func(net *gofann.Fann[float32]) {
				// Uses default RPROP parameters
			},
		},
		{
			name: "Quickprop",
			algo: gofann.TrainQuickprop,
			setup: func(net *gofann.Fann[float32]) {
				net.SetLearningRate(0.7)
				net.SetQuickpropDecay(-0.0001)
				net.SetQuickpropMu(1.75)
			},
		},
		{
			name: "Sarprop",
			algo: gofann.TrainSarprop,
			setup: func(net *gofann.Fann[float32]) {
				// Uses default Sarprop parameters
			},
		},
	}
	
	for _, alg := range algorithms {
		fmt.Printf("\n%s Training:\n", alg.name)
		fmt.Println("-------------------")
		
		// Create fresh network
		net := gofann.CreateStandard[float32](2, 4, 1)
		net.SetTrainingAlgorithm(alg.algo)
		alg.setup(net)
		
		// Track progress
		epochHistory := []float32{}
		net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
			epochHistory = append(epochHistory, mse)
			return true
		})
		
		// Time the training
		start := time.Now()
		net.TrainOnData(data, 1000, 100, 0.001)
		duration := time.Since(start)
		
		// Show progress
		fmt.Printf("Training time: %v\n", duration)
		fmt.Printf("Progress: ")
		for i, mse := range epochHistory {
			fmt.Printf("Epoch %d: %.4f  ", (i+1)*100, mse)
		}
		fmt.Println()
		
		// Test final network
		fmt.Println("Final results:")
		finalMSE := float32(0)
		for i := 0; i < data.GetNumData(); i++ {
			input := data.GetInput(i)
			output := net.Run(input)
			expected := data.GetOutput(i)
			
			diff := expected[0] - output[0]
			finalMSE += diff * diff
			
			fmt.Printf("  %v -> %.3f (expected %.1f)\n", 
				input, output[0], expected[0])
		}
		finalMSE /= float32(data.GetNumData())
		fmt.Printf("Final MSE: %.6f\n", finalMSE)
	}
	
	fmt.Println("\nNote: Results may vary due to random initialization.")
}