package gofann

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	// Create a test network
	net1 := CreateStandard[float32](2, 3, 1)
	if net1 == nil {
		t.Fatal("Failed to create network")
	}
	
	// Set some parameters
	net1.SetLearningRate(0.5)
	net1.SetLearningMomentum(0.1)
	net1.SetTrainingAlgorithm(TrainBatch)
	
	// Train it a bit on XOR
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	trainData := CreateTrainDataArray(inputs, outputs)
	
	net1.TrainOnData(trainData, 100, 0, 0.1)
	
	// Get output before saving
	testInput := []float32{0, 1}
	output1 := net1.Run(testInput)
	
	// Save network
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "test_network.net")
	
	err := net1.Save(filename)
	if err != nil {
		t.Fatalf("Failed to save network: %v", err)
	}
	
	// Check file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Fatal("Save file was not created")
	}
	
	// Load network
	net2, err := CreateFromFile[float32](filename)
	if err != nil {
		t.Fatalf("Failed to load network: %v", err)
	}
	
	// Compare properties
	if net2.GetNumInput() != net1.GetNumInput() {
		t.Errorf("NumInput mismatch: got %d, want %d", net2.GetNumInput(), net1.GetNumInput())
	}
	if net2.GetNumOutput() != net1.GetNumOutput() {
		t.Errorf("NumOutput mismatch: got %d, want %d", net2.GetNumOutput(), net1.GetNumOutput())
	}
	if net2.GetTotalNeurons() != net1.GetTotalNeurons() {
		t.Errorf("TotalNeurons mismatch: got %d, want %d", net2.GetTotalNeurons(), net1.GetTotalNeurons())
	}
	if net2.GetLearningRate() != net1.GetLearningRate() {
		t.Errorf("LearningRate mismatch: got %f, want %f", net2.GetLearningRate(), net1.GetLearningRate())
	}
	if net2.GetTrainingAlgorithm() != net1.GetTrainingAlgorithm() {
		t.Errorf("TrainingAlgorithm mismatch: got %v, want %v", net2.GetTrainingAlgorithm(), net1.GetTrainingAlgorithm())
	}
	
	// Test that loaded network produces same output
	output2 := net2.Run(testInput)
	if len(output1) != len(output2) {
		t.Fatalf("Output length mismatch: got %d, want %d", len(output2), len(output1))
	}
	
	diff := abs(output1[0] - output2[0])
	if diff > 0.01 { // Allow 1% difference due to float32 precision in file format
		t.Errorf("Output mismatch: got %f, want %f (diff: %f)", output2[0], output1[0], diff)
	}
	
	// Test network still works after loading
	output3 := net2.Run([]float32{1, 1})
	t.Logf("Loaded network output for [1,1]: %f", output3[0])
}

func TestSaveLoadLargerNetwork(t *testing.T) {
	// Test with a larger network
	net1 := CreateStandard[float64](10, 20, 15, 5)
	if net1 == nil {
		t.Fatal("Failed to create network")
	}
	
	// Save and load
	tmpDir := t.TempDir()
	filename := filepath.Join(tmpDir, "large_network.net")
	
	err := net1.Save(filename)
	if err != nil {
		t.Fatalf("Failed to save network: %v", err)
	}
	
	net2, err := CreateFromFile[float64](filename)
	if err != nil {
		t.Fatalf("Failed to load network: %v", err)
	}
	
	// Compare layer sizes
	sizes1 := net1.GetLayerSizes()
	sizes2 := net2.GetLayerSizes()
	
	if len(sizes1) != len(sizes2) {
		t.Fatalf("Layer count mismatch: got %d, want %d", len(sizes2), len(sizes1))
	}
	
	for i := range sizes1 {
		if sizes1[i] != sizes2[i] {
			t.Errorf("Layer %d size mismatch: got %d, want %d", i, sizes2[i], sizes1[i])
		}
	}
	
	// Test with random input
	testInput := make([]float64, 10)
	for i := range testInput {
		testInput[i] = float64(i) * 0.1
	}
	
	output1 := net1.Run(testInput)
	output2 := net2.Run(testInput)
	
	if len(output1) != len(output2) {
		t.Fatalf("Output length mismatch: got %d, want %d", len(output2), len(output1))
	}
	
	for i := range output1 {
		diff := abs(output1[i] - output2[i])
		if diff > 0.0001 {
			t.Errorf("Output[%d] mismatch: got %f, want %f", i, output2[i], output1[i])
		}
	}
}