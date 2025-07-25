// Sarprop demonstration - shows the benefit of weight decay
package main

import (
	"fmt"
	"strings"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN Sarprop Demonstration")
	fmt.Println("============================")
	fmt.Println("Sarprop includes weight decay to prevent overfitting")
	fmt.Println()
	
	// Load diabetes dataset - good for demonstrating regularization
	trainPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/diabetes.train"
	testPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/diabetes.test"
	
	trainData, err := gofann.ReadTrainFromFile[float32](trainPath)
	if err != nil {
		fmt.Printf("Error loading training data: %v\n", err)
		return
	}
	
	testData, err := gofann.ReadTrainFromFile[float32](testPath)
	if err != nil {
		fmt.Printf("Error loading test data: %v\n", err)
		return
	}
	
	fmt.Printf("Loaded diabetes dataset: %d training, %d test samples\n",
		trainData.GetNumData(), testData.GetNumData())
	fmt.Printf("Inputs: %d, Outputs: %d\n", 
		trainData.GetNumInput(), trainData.GetNumOutput())
	
	// Compare RPROP vs Sarprop
	algorithms := []struct {
		name  string
		algo  gofann.TrainAlgorithm
		setup func(net *gofann.Fann[float32])
	}{
		{
			name: "RPROP (no weight decay)",
			algo: gofann.TrainRPROP,
			setup: func(net *gofann.Fann[float32]) {},
		},
		{
			name: "Sarprop (with weight decay)",
			algo: gofann.TrainSarprop,
			setup: func(net *gofann.Fann[float32]) {
				// Default parameters are good
			},
		},
		{
			name: "Sarprop (stronger decay)",
			algo: gofann.TrainSarprop,
			setup: func(net *gofann.Fann[float32]) {
				net.SetSarpropWeightDecayShift(-5.0) // Stronger weight decay
			},
		},
	}
	
	for _, alg := range algorithms {
		fmt.Printf("\n%s:\n", alg.name)
		fmt.Println(strings.Repeat("-", len(alg.name)+1))
		
		// Create network with more hidden neurons (prone to overfitting)
		net := gofann.CreateStandard[float32](8, 20, 15, 2)
		net.SetTrainingAlgorithm(alg.algo)
		alg.setup(net)
		
		// Track training and test MSE
		trainMSEs := []float32{}
		testMSEs := []float32{}
		
		// Test every 10 epochs
		for epoch := 0; epoch <= 100; epoch += 10 {
			if epoch > 0 {
				// Train for 10 epochs
				for i := 0; i < 10; i++ {
					net.TrainEpoch(trainData)
				}
			}
			
			trainMSE := net.TestData(trainData)
			testMSE := net.TestData(testData)
			
			trainMSEs = append(trainMSEs, trainMSE)
			testMSEs = append(testMSEs, testMSE)
			
			if epoch%20 == 0 {
				fmt.Printf("Epoch %3d: Train MSE=%.4f, Test MSE=%.4f\n",
					epoch, trainMSE, testMSE)
			}
		}
		
		// Check for overfitting
		finalTrainMSE := trainMSEs[len(trainMSEs)-1]
		finalTestMSE := testMSEs[len(testMSEs)-1]
		overfitting := finalTestMSE - finalTrainMSE
		
		fmt.Printf("\nFinal: Train MSE=%.4f, Test MSE=%.4f\n", 
			finalTrainMSE, finalTestMSE)
		fmt.Printf("Overfitting indicator (test-train): %.4f\n", overfitting)
		
		// Test classification accuracy
		correct := 0
		for i := 0; i < testData.GetNumData(); i++ {
			input := testData.GetInput(i)
			output := net.Run(input)
			expected := testData.GetOutput(i)
			
			// Find max output
			predictedClass := 0
			if output[1] > output[0] {
				predictedClass = 1
			}
			
			expectedClass := 0
			if expected[1] > expected[0] {
				expectedClass = 1
			}
			
			if predictedClass == expectedClass {
				correct++
			}
		}
		
		accuracy := float32(correct) / float32(testData.GetNumData()) * 100
		fmt.Printf("Test accuracy: %.1f%%\n", accuracy)
	}
	
	fmt.Println("\nNote: Sarprop's weight decay helps prevent overfitting,")
	fmt.Println("especially important with larger networks.")
}

