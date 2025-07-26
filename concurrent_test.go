package gofann

import (
	"fmt"
	"testing"
	"time"
)

// TestConcurrentExpertTraining tests parallel training of multiple experts
func TestConcurrentExpertTraining(t *testing.T) {
	// Create training data for different patterns
	xorData := CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {1}, {1}, {0}},
	)
	
	andData := CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {0}, {0}, {1}},
	)
	
	orData := CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {1}, {1}, {1}},
	)
	
	// Create experts
	experts := []*ReflectiveExpert[float32]{
		NewReflectiveExpert[float32]("XORExpert", "xor", []int{2, 4, 1}),
		NewReflectiveExpert[float32]("ANDExpert", "and", []int{2, 3, 1}),
		NewReflectiveExpert[float32]("ORExpert", "or", []int{2, 3, 1}),
	}
	
	// Initialize networks
	for _, expert := range experts {
		expert.network.RandomizeWeights(-1, 1)
		expert.network.SetLearningRate(0.7)
		expert.network.SetTrainingAlgorithm(TrainRPROP)
	}
	
	// Prepare data map
	dataMap := map[string]*TrainData[float32]{
		"xor": xorData,
		"and": andData,
		"or":  orData,
	}
	
	// Create concurrent trainer
	trainer := NewConcurrentTrainer[float32](3)
	
	// Start timing
	start := time.Now()
	
	// Train all experts concurrently
	results := trainer.TrainExperts(experts, dataMap)
	
	duration := time.Since(start)
	
	// Verify results
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
	
	t.Logf("✅ Concurrent training completed in %v", duration)
	t.Logf("Results:")
	for _, result := range results {
		t.Logf("  %s: MSE=%.6f, Epochs=%d, Success=%v",
			result.ID, result.FinalMSE, result.EpochsTrained, result.Success)
	}
	
	// Test the trained experts
	testCases := []struct {
		expert   *ReflectiveExpert[float32]
		input    []float32
		expected float32
		name     string
	}{
		{experts[0], []float32{0, 0}, 0, "XOR(0,0)"},
		{experts[0], []float32{1, 1}, 0, "XOR(1,1)"},
		{experts[1], []float32{1, 1}, 1, "AND(1,1)"},
		{experts[1], []float32{0, 1}, 0, "AND(0,1)"},
		{experts[2], []float32{0, 0}, 0, "OR(0,0)"},
		{experts[2], []float32{1, 0}, 1, "OR(1,0)"},
	}
	
	for _, tc := range testCases {
		output := tc.expert.network.Run(tc.input)
		if len(output) > 0 {
			t.Logf("  %s = %.3f (expected %.0f)", tc.name, output[0], tc.expected)
		}
	}
}

// TestParallelBatchProcessing tests concurrent batch inference
func TestParallelBatchProcessing(t *testing.T) {
	// Create a trained network
	net := CreateStandard[float32](2, 4, 1)
	net.RandomizeWeights(-1, 1)
	
	// Create batch processor
	processor := NewBatchProcessor(net, 4)
	
	// Create large batch of inputs
	batchSize := 1000
	inputs := make([][]float32, batchSize)
	for i := 0; i < batchSize; i++ {
		inputs[i] = []float32{float32(i % 2), float32((i / 2) % 2)}
	}
	
	// Time sequential processing
	start := time.Now()
	seqOutputs := make([][]float32, batchSize)
	for i, input := range inputs {
		seqOutputs[i] = net.Run(input)
	}
	seqDuration := time.Since(start)
	
	// Time parallel processing
	start = time.Now()
	parOutputs := processor.ProcessBatch(inputs)
	parDuration := time.Since(start)
	
	// Verify outputs match
	for i := range inputs {
		if len(seqOutputs[i]) != len(parOutputs[i]) {
			t.Errorf("Output length mismatch at index %d", i)
			continue
		}
		for j := range seqOutputs[i] {
			diff := seqOutputs[i][j] - parOutputs[i][j]
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.0001 {
				t.Errorf("Output mismatch at [%d][%d]: seq=%.6f, par=%.6f",
					i, j, seqOutputs[i][j], parOutputs[i][j])
			}
		}
	}
	
	speedup := float64(seqDuration) / float64(parDuration)
	t.Logf("✅ Batch processing verified")
	t.Logf("  Sequential: %v", seqDuration)
	t.Logf("  Parallel:   %v", parDuration)
	t.Logf("  Speedup:    %.2fx", speedup)
}

// TestTrainingMonitor tests the visualization system
func TestTrainingMonitor(t *testing.T) {
	// Create a simple network
	net := CreateStandard[float32](2, 3, 1)
	net.RandomizeWeights(-1, 1)
	
	// Create monitor
	monitor := NewTrainingMonitor[float32]()
	
	// Simulate training with monitoring
	data := CreateTrainDataArray[float32](
		[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
		[][]float32{{0}, {1}, {1}, {0}},
	)
	
	totalEpochs := 100
	monitor.Start(totalEpochs)
	
	// Set callback to update monitor
	net.SetCallback(func(ann *Fann[float32], epoch int, mse float32) bool {
		// Calculate accuracy (simplified)
		correct := 0
		for i := 0; i < data.GetNumData(); i++ {
			output := ann.Run(data.GetInput(i))
			expected := data.GetOutput(i)
			if len(output) > 0 && len(expected) > 0 {
				diff := output[0] - expected[0]
				if diff < 0 {
					diff = -diff
				}
				if diff < 0.5 {
					correct++
				}
			}
		}
		accuracy := float32(correct) / float32(data.GetNumData())
		
		monitor.Update(epoch, mse, accuracy, ann.GetLearningRate())
		return true
	})
	
	// Quick training run
	net.TrainOnData(data, totalEpochs, 10, 0.01)
	
	monitor.Complete()
	
	t.Logf("✅ Training monitor test completed")
}

// TestConcurrentTrainingWithMonitoring combines concurrent training with monitoring
func TestConcurrentTrainingWithMonitoring(t *testing.T) {
	// Create expert monitor
	monitor := NewExpertMonitor[float32]()
	
	expertNames := []string{"XORMaster", "ANDWizard", "ORGuru"}
	monitor.Start(expertNames)
	
	// Simulate expert training updates
	go func() {
		for i := 0; i < 10; i++ {
			for _, name := range expertNames {
				monitor.Update(ExpertUpdate[float32]{
					Name:     name,
					Epoch:    i * 100,
					MSE:      0.5 / float32(i+1),
					Accuracy: float32(i) / 10.0,
					Status:   "training",
				})
			}
			time.Sleep(100 * time.Millisecond)
		}
		
		// Mark as completed
		for _, name := range expertNames {
			monitor.Update(ExpertUpdate[float32]{
				Name:     name,
				Epoch:    1000,
				MSE:      0.01,
				Accuracy: 0.99,
				Status:   "completed",
			})
		}
	}()
	
	// Let it run for a bit
	time.Sleep(1200 * time.Millisecond)
	
	monitor.Stop()
	
	t.Logf("✅ Multi-expert monitoring test completed")
}

// BenchmarkConcurrentTraining benchmarks parallel vs sequential training
func BenchmarkConcurrentTraining(b *testing.B) {
	// Create multiple datasets
	numExperts := 8
	datasets := make([]*TrainData[float32], numExperts)
	for i := 0; i < numExperts; i++ {
		datasets[i] = CreateTrainDataArray[float32](
			[][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}},
			[][]float32{{0}, {1}, {1}, {0}},
		)
	}
	
	b.Run("Sequential", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			for i := 0; i < numExperts; i++ {
				net := CreateStandard[float32](2, 4, 1)
				net.TrainOnData(datasets[i], 100, 0, 0.01)
			}
		}
	})
	
	b.Run("Concurrent-4", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			trainer := NewConcurrentTrainer[float32](4)
			batches := make([]*ConcurrentTrainBatch[float32], numExperts)
			
			for i := 0; i < numExperts; i++ {
				batches[i] = &ConcurrentTrainBatch[float32]{
					Network:      CreateStandard[float32](2, 4, 1),
					Data:         datasets[i],
					MaxEpochs:    100,
					DesiredError: 0.01,
					ID:           fmt.Sprintf("Expert%d", i),
				}
			}
			
			trainer.TrainBatches(batches)
		}
	})
}