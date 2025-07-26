// Package gofann - Concurrent Training Support
// Enables parallel training of multiple experts and batch processing
package gofann

import (
	"fmt"
	"runtime"
	"sync"
)

// ConcurrentTrainer manages parallel training of multiple networks
type ConcurrentTrainer[T Numeric] struct {
	workers    int
	batchQueue chan *ConcurrentTrainBatch[T]
	wg         sync.WaitGroup
	mu         sync.Mutex
	results    []TrainResult[T]
}

// ConcurrentTrainBatch represents a batch of training work for concurrent processing
type ConcurrentTrainBatch[T Numeric] struct {
	Network      *Fann[T]
	Data         *TrainData[T]
	MaxEpochs    int
	DesiredError float32
	ID           string
}

// TrainResult captures the outcome of training
type TrainResult[T Numeric] struct {
	ID           string
	FinalMSE     float32
	EpochsTrained int
	Success      bool
	Error        error
}

// NewConcurrentTrainer creates a trainer with specified worker count
func NewConcurrentTrainer[T Numeric](workers int) *ConcurrentTrainer[T] {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	return &ConcurrentTrainer[T]{
		workers:    workers,
		batchQueue: make(chan *ConcurrentTrainBatch[T], workers*2),
		results:    make([]TrainResult[T], 0),
	}
}

// TrainExperts trains multiple experts concurrently
func (ct *ConcurrentTrainer[T]) TrainExperts(experts []*ReflectiveExpert[T], data map[string]*TrainData[T]) []TrainResult[T] {
	fmt.Printf("🚀 Starting concurrent training of %d experts with %d workers\n", len(experts), ct.workers)
	
	// Start worker pool
	ct.startWorkers()
	
	// Queue training jobs
	for _, expert := range experts {
		if trainData, ok := data[expert.domain]; ok {
			batch := &ConcurrentTrainBatch[T]{
				Network:      expert.network,
				Data:         trainData,
				MaxEpochs:    1000,
				DesiredError: 0.01,
				ID:           expert.name,
			}
			ct.batchQueue <- batch
		}
	}
	
	// Close queue and wait for completion
	close(ct.batchQueue)
	ct.wg.Wait()
	
	fmt.Printf("✅ Concurrent training complete! Trained %d experts\n", len(ct.results))
	return ct.results
}

// TrainBatches trains multiple networks with different data concurrently
func (ct *ConcurrentTrainer[T]) TrainBatches(batches []*ConcurrentTrainBatch[T]) []TrainResult[T] {
	fmt.Printf("🚀 Starting concurrent batch training with %d batches\n", len(batches))
	
	// Start worker pool
	ct.startWorkers()
	
	// Queue all batches
	for _, batch := range batches {
		ct.batchQueue <- batch
	}
	
	// Close queue and wait
	close(ct.batchQueue)
	ct.wg.Wait()
	
	return ct.results
}

// startWorkers launches the worker goroutines
func (ct *ConcurrentTrainer[T]) startWorkers() {
	for i := 0; i < ct.workers; i++ {
		ct.wg.Add(1)
		go ct.worker(i)
	}
}

// worker processes training batches
func (ct *ConcurrentTrainer[T]) worker(id int) {
	defer ct.wg.Done()
	
	for batch := range ct.batchQueue {
		result := ct.trainSingle(batch)
		
		// Store result thread-safely
		ct.mu.Lock()
		ct.results = append(ct.results, result)
		ct.mu.Unlock()
		
		if result.Success {
			fmt.Printf("   Worker %d: ✓ %s trained (MSE: %.6f)\n", 
				id, result.ID, result.FinalMSE)
		} else {
			fmt.Printf("   Worker %d: ✗ %s failed: %v\n", 
				id, result.ID, result.Error)
		}
	}
}

// trainSingle trains a single network
func (ct *ConcurrentTrainer[T]) trainSingle(batch *ConcurrentTrainBatch[T]) TrainResult[T] {
	result := TrainResult[T]{
		ID: batch.ID,
	}
	
	// Reset MSE before training
	batch.Network.ResetMSE()
	
	// Track epochs
	epochsTrained := 0
	callback := func(ann *Fann[T], epochs int, mse float32) bool {
		epochsTrained = epochs
		return true // Continue training
	}
	batch.Network.SetCallback(callback)
	
	// Perform training
	batch.Network.TrainOnData(batch.Data, batch.MaxEpochs, 10, batch.DesiredError)
	
	// Test final MSE on all data
	finalMSE := batch.Network.TestData(batch.Data)
	
	// Capture results
	result.FinalMSE = finalMSE
	result.EpochsTrained = epochsTrained
	result.Success = result.FinalMSE <= batch.DesiredError
	
	return result
}

// ParallelEpochTrainer enables data-parallel training within a single network
type ParallelEpochTrainer[T Numeric] struct {
	network    *Fann[T]
	workers    int
	subsetSize int
}

// NewParallelEpochTrainer creates a trainer for data-parallel epochs
func NewParallelEpochTrainer[T Numeric](network *Fann[T], workers int) *ParallelEpochTrainer[T] {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	return &ParallelEpochTrainer[T]{
		network: network,
		workers: workers,
	}
}

// TrainEpochParallel trains one epoch using data parallelism
func (pet *ParallelEpochTrainer[T]) TrainEpochParallel(data *TrainData[T]) float32 {
	numData := data.GetNumData()
	pet.subsetSize = (numData + pet.workers - 1) / pet.workers
	
	// Create weight update accumulators
	gradients := make([][]T, pet.workers)
	mseSums := make([]float64, pet.workers)
	var wg sync.WaitGroup
	
	// Process subsets in parallel
	for w := 0; w < pet.workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * pet.subsetSize
			end := start + pet.subsetSize
			if end > numData {
				end = numData
			}
			
			// Create subset
			subset := data.Subset(start, end-start)
			if subset == nil {
				return
			}
			
			// Clone network for this worker
			workerNet := pet.network.Copy()
			
			// Train on subset
			workerNet.ResetMSE()
			for i := 0; i < subset.GetNumData(); i++ {
				workerNet.Train(subset.GetInput(i), subset.GetOutput(i))
			}
			
			// Store gradients and MSE
			gradients[workerID] = workerNet.GetWeights()
			mseSums[workerID] = float64(workerNet.GetMSE()) * float64(subset.GetNumData())
		}(w)
	}
	
	wg.Wait()
	
	// Average gradients and update main network
	pet.averageAndApplyGradients(gradients)
	
	// Calculate total MSE
	totalMSE := float64(0)
	for _, mse := range mseSums {
		totalMSE += mse
	}
	
	return float32(totalMSE / float64(numData))
}

// averageAndApplyGradients combines gradients from all workers
func (pet *ParallelEpochTrainer[T]) averageAndApplyGradients(gradients [][]T) {
	if len(gradients) == 0 {
		return
	}
	
	// Get current weights
	weights := pet.network.GetWeights()
	
	// Average the gradients
	for i := range weights {
		sum := T(0)
		for w := range gradients {
			if i < len(gradients[w]) {
				sum += gradients[w][i]
			}
		}
		weights[i] = sum / T(len(gradients))
	}
	
	// Apply averaged weights
	pet.network.SetWeights(weights)
}

// BatchProcessor enables efficient batch inference
type BatchProcessor[T Numeric] struct {
	network *Fann[T]
	workers int
}

// NewBatchProcessor creates a processor for parallel inference
func NewBatchProcessor[T Numeric](network *Fann[T], workers int) *BatchProcessor[T] {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	return &BatchProcessor[T]{
		network: network,
		workers: workers,
	}
}

// ProcessBatch runs inference on multiple inputs concurrently
func (bp *BatchProcessor[T]) ProcessBatch(inputs [][]T) [][]T {
	outputs := make([][]T, len(inputs))
	
	// Use worker pool for large batches
	if len(inputs) > bp.workers*10 {
		var wg sync.WaitGroup
		chunkSize := (len(inputs) + bp.workers - 1) / bp.workers
		
		for w := 0; w < bp.workers; w++ {
			wg.Add(1)
			start := w * chunkSize
			end := start + chunkSize
			if end > len(inputs) {
				end = len(inputs)
			}
			
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					outputs[i] = bp.network.Run(inputs[i])
				}
			}(start, end)
		}
		
		wg.Wait()
	} else {
		// Process sequentially for small batches
		for i, input := range inputs {
			outputs[i] = bp.network.Run(input)
		}
	}
	
	return outputs
}