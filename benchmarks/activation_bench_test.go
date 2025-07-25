package benchmarks

import (
	"testing"
	"github.com/lynncoleart/gofann"
)

// Benchmark all activation functions for float32
func BenchmarkActivationFloat32(b *testing.B) {
	activations := []struct {
		name string
		fn   gofann.ActivationFunc
	}{
		{"Linear", gofann.Linear},
		{"Sigmoid", gofann.Sigmoid},
		{"SigmoidSymmetric", gofann.SigmoidSymmetric},
		{"Gaussian", gofann.Gaussian},
		{"Elliot", gofann.Elliot},
		{"Sin", gofann.Sin},
		{"Threshold", gofann.Threshold},
	}

	steepness := float32(0.5)
	value := float32(0.7)

	for _, act := range activations {
		b.Run(act.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofann.Activation(act.fn, steepness, value)
			}
		})
	}
}

// Benchmark all activation functions for float64
func BenchmarkActivationFloat64(b *testing.B) {
	activations := []struct {
		name string
		fn   gofann.ActivationFunc
	}{
		{"Linear", gofann.Linear},
		{"Sigmoid", gofann.Sigmoid},
		{"SigmoidSymmetric", gofann.SigmoidSymmetric},
		{"Gaussian", gofann.Gaussian},
		{"Elliot", gofann.Elliot},
		{"Sin", gofann.Sin},
		{"Threshold", gofann.Threshold},
	}

	steepness := float64(0.5)
	value := float64(0.7)

	for _, act := range activations {
		b.Run(act.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofann.Activation(act.fn, steepness, value)
			}
		})
	}
}

// Benchmark vectorized activation (simulating SIMD-like operation)
func BenchmarkActivationVectorized(b *testing.B) {
	// Test processing 8 values at once (AVX2-like)
	values := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	steepness := float32(0.5)
	results := make([]float32, 8)

	b.Run("Sigmoid_8x", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 8; j++ {
				results[j] = gofann.Activation(gofann.Sigmoid, steepness, values[j])
			}
		}
	})

	b.Run("Linear_8x", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for j := 0; j < 8; j++ {
				results[j] = gofann.Activation(gofann.Linear, steepness, values[j])
			}
		}
	})
}

// Benchmark activation derivatives
func BenchmarkActivationDerivative(b *testing.B) {
	activations := []struct {
		name string
		fn   gofann.ActivationFunc
	}{
		{"Linear", gofann.Linear},
		{"Sigmoid", gofann.Sigmoid},
		{"SigmoidSymmetric", gofann.SigmoidSymmetric},
		{"Gaussian", gofann.Gaussian},
		{"Elliot", gofann.Elliot},
	}

	steepness := float32(0.5)
	value := float32(0.7)
	sum := float32(0.3)

	for _, act := range activations {
		b.Run(act.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = gofann.ActivationDerivative(act.fn, steepness, value, sum)
			}
		})
	}
}