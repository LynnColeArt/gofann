// Mushroom classifier example using FANN dataset
package main

import (
	"fmt"
	"log"
	"github.com/lynncoleart/gofann"
)

func main() {
	fmt.Println("GoFANN Mushroom Classifier Example")
	fmt.Println("==================================")
	
	// Load the mushroom training data
	trainPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/mushroom.train"
	testPath := "/media/lynn/big_drive/workspaces/fanmaker/fann_source/datasets/mushroom.test"
	
	// Load training data
	trainData, err := gofann.ReadTrainFromFile[float32](trainPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	
	fmt.Printf("Loaded training data: %d samples, %d inputs, %d outputs\n",
		trainData.GetNumData(), trainData.GetNumInput(), trainData.GetNumOutput())
	
	// Create network
	// Input: 125 binary features
	// Hidden layers: 30 neurons
	// Output: 2 (one-hot encoded: edible or poisonous)
	net := gofann.CreateStandard[float32](125, 30, 2)
	if net == nil {
		log.Fatal("Failed to create network")
	}
	
	// Configure training
	net.SetTrainingAlgorithm(gofann.TrainIncremental)
	net.SetLearningRate(0.1)
	net.SetLearningMomentum(0.1)
	
	// Test before training
	testData, err := gofann.ReadTrainFromFile[float32](testPath)
	if err != nil {
		log.Fatalf("Failed to load test data: %v", err)
	}
	
	initialMSE := net.TestData(testData)
	fmt.Printf("\nInitial MSE on test set: %.4f\n", initialMSE)
	
	// Count initial accuracy
	correct := 0
	for i := 0; i < testData.GetNumData() && i < 10; i++ {
		input := testData.GetInput(i)
		output := net.Run(input)
		expected := testData.GetOutput(i)
		
		// Determine predicted class
		predictedClass := 0
		if output[1] > output[0] {
			predictedClass = 1
		}
		
		// Determine expected class
		expectedClass := 0
		if expected[1] > expected[0] {
			expectedClass = 1
		}
		
		if predictedClass == expectedClass {
			correct++
		}
	}
	fmt.Printf("Initial accuracy (first 10 samples): %d/10\n", correct)
	
	// Set up training callback
	net.SetCallback(func(ann *gofann.Fann[float32], epochs int, mse float32) bool {
		fmt.Printf("Epoch %5d - Training MSE: %.6f\n", epochs, mse)
		return true
	})
	
	// Train the network
	fmt.Println("\nTraining network...")
	net.TrainOnData(trainData, 10, 2, 0.001)
	
	// Test after training
	finalMSE := net.TestData(testData)
	fmt.Printf("\nFinal MSE on test set: %.4f\n", finalMSE)
	
	// Calculate accuracy on full test set
	correct = 0
	for i := 0; i < testData.GetNumData(); i++ {
		input := testData.GetInput(i)
		output := net.Run(input)
		expected := testData.GetOutput(i)
		
		// Determine predicted class
		predictedClass := 0
		if output[1] > output[0] {
			predictedClass = 1
		}
		
		// Determine expected class
		expectedClass := 0
		if expected[1] > expected[0] {
			expectedClass = 1
		}
		
		if predictedClass == expectedClass {
			correct++
		}
	}
	
	accuracy := float64(correct) / float64(testData.GetNumData()) * 100.0
	fmt.Printf("Final accuracy on test set: %.2f%% (%d/%d correct)\n", 
		accuracy, correct, testData.GetNumData())
	
	// Show some example predictions
	fmt.Println("\nExample predictions (first 5 test samples):")
	for i := 0; i < 5 && i < testData.GetNumData(); i++ {
		input := testData.GetInput(i)
		output := net.Run(input)
		expected := testData.GetOutput(i)
		
		prediction := "Edible"
		if output[1] > output[0] {
			prediction = "Poisonous"
		}
		
		actual := "Edible"
		if expected[1] > 0.5 {
			actual = "Poisonous"
		}
		
		fmt.Printf("  Sample %d: Predicted=%s (%.3f, %.3f), Actual=%s\n",
			i+1, prediction, output[0], output[1], actual)
	}
	
	// Save the trained network
	fmt.Println("\nSaving trained network...")
	err = net.Save("mushroom_classifier.net")
	if err != nil {
		log.Printf("Failed to save network: %v", err)
	} else {
		fmt.Println("Network saved to mushroom_classifier.net")
	}
}