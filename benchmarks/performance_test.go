package benchmarks

import (
	"testing"
	"github.com/LynnColeArt/gofann"
)

// BenchmarkXORTraining benchmarks XOR training performance
func BenchmarkXORTraining(b *testing.B) {
	// XOR training data
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	
	b.Run("RPROP", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			net := gofann.CreateStandard[float32](2, 4, 1)
			net.SetTrainingAlgorithm(gofann.TrainRPROP)
			net.SetLearningRate(0.7)
			
			trainData := gofann.CreateTrainDataArray(inputs, outputs)
			net.TrainOnData(trainData, 1000, 0, 0.001)
		}
	})
	
	b.Run("Batch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			net := gofann.CreateStandard[float32](2, 4, 1)
			net.SetTrainingAlgorithm(gofann.TrainBatch)
			net.SetLearningRate(0.7)
			
			trainData := gofann.CreateTrainDataArray(inputs, outputs)
			net.TrainOnData(trainData, 1000, 0, 0.001)
		}
	})
	
	b.Run("Quickprop", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			net := gofann.CreateStandard[float32](2, 4, 1)
			net.SetTrainingAlgorithm(gofann.TrainQuickprop)
			net.SetLearningRate(0.7)
			
			trainData := gofann.CreateTrainDataArray(inputs, outputs)
			net.TrainOnData(trainData, 1000, 0, 0.001)
		}
	})
}

// BenchmarkNetworkRun benchmarks forward propagation
func BenchmarkNetworkRun(b *testing.B) {
	sizes := [][]int{
		{2, 4, 1},      // Small
		{10, 20, 5},    // Medium  
		{50, 100, 20},  // Large
	}
	
	for _, size := range sizes {
		name := fmt.Sprintf("%dx%dx%d", size[0], size[1], size[2])
		b.Run(name, func(b *testing.B) {
			net := gofann.CreateStandard[float32](size...)
			input := make([]float32, size[0])
			for i := range input {
				input[i] = 0.5
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				net.Run(input)
			}
		})
	}
}

// BenchmarkActivationFunctions benchmarks different activation functions
func BenchmarkActivationFunctions(b *testing.B) {
	functions := []struct{
		name string
		fn   gofann.ActivationFunc
	}{
		{"Sigmoid", gofann.Sigmoid},
		{"SigmoidSymmetric", gofann.SigmoidSymmetric}, 
		{"Gaussian", gofann.Gaussian},
		{"Linear", gofann.Linear},
		{"Elliot", gofann.Elliot},
	}
	
	net := gofann.CreateStandard[float32](10, 20, 5)
	input := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	
	for _, fn := range functions {
		b.Run(fn.name, func(b *testing.B) {
			net.SetActivationFunctionHidden(fn.fn)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				net.Run(input)
			}
		})
	}
}

// BenchmarkCascadeTraining benchmarks cascade training
func BenchmarkCascadeTraining(b *testing.B) {
	inputs := [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	outputs := [][]float32{{0}, {1}, {1}, {0}}
	trainData := gofann.CreateTrainDataArray(inputs, outputs)
	
	b.Run("Cascade", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			net := gofann.CreateCascade[float32](2, 1)
			net.CascadetrainOnData(trainData, 10, 0, 0.001)
		}
	})
}