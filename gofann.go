// Package gofann implements a pure Go port of the Fast Artificial Neural Network (FANN) library.
// It provides a simple, fast, and effective way to create and train neural networks.
package gofann

import (
	"fmt"
)

// Version is the GoFANN version, compatible with FANN 2.3.0
const Version = "2.3.0-go"

// Numeric is the constraint for numeric types used in the network
type Numeric interface {
	~float32 | ~float64
}

// ActivationFunc represents the activation function types
type ActivationFunc int

const (
	Linear ActivationFunc = iota
	Threshold
	ThresholdSymmetric
	Sigmoid
	SigmoidStepwise
	SigmoidSymmetric
	SigmoidSymmetricStepwise
	Gaussian
	GaussianSymmetric
	GaussianStepwise
	Elliot
	ElliotSymmetric
	LinearPiece
	LinearPieceSymmetric
	SinSymmetric
	CosSymmetric
	Sin
	Cos
	LinearPieceRect
	LinearPieceRectLeaky
)

// TrainAlgorithm represents training algorithms
type TrainAlgorithm int

const (
	TrainIncremental TrainAlgorithm = iota
	TrainBatch
	TrainRPROP
	TrainQuickprop
	TrainSarprop
)

// ErrorFunc represents error functions
type ErrorFunc int

const (
	ErrorLinear ErrorFunc = iota
	ErrorTanh
)

// NetworkType represents network topology types
type NetworkType int

const (
	NetTypeLayer    NetworkType = iota // Each layer only has connections to the next layer
	NetTypeShortcut                    // Each layer has connections to all following layers
)

// ErrorCode represents FANN error codes
type ErrorCode int

const (
	ErrNoError ErrorCode = iota
	ErrCantOpenConfig
	ErrCantOpenTD
	ErrCantReadConfig
	ErrCantReadTD
	ErrWrongConfigVersion
	ErrWrongTDVersion
	ErrCantReadNeuron
	ErrCantReadConnection
	ErrWrongNumConnections
	ErrCantOpenWriter
	ErrCantWriteConfig
	ErrCantWriteTD
	ErrCantAllocateMem
	ErrCantTrainActivation
	ErrCantUseActivation
	ErrTrainDataMismatch
	ErrCantUseTrainAlg
	ErrTrainDataSubset
	ErrIndexOutOfBound
	ErrScaleNotPresent
	ErrInputMismatch
	ErrOutputMismatch
	ErrWrongParameters
)

// neuron represents a single neuron
type neuron[T Numeric] struct {
	firstCon            int            // Index to first connection
	lastCon             int            // Index to last connection (exclusive)
	sum                 T              // Weighted sum of inputs
	value               T              // Output value after activation
	activationSteepness T              // Steepness parameter for activation
	activationFunction  ActivationFunc // Which activation function to use
}

// layer represents a layer of neurons
type layer[T Numeric] struct {
	firstNeuron int // Index into neurons slice
	lastNeuron  int // Index past last neuron (exclusive)
}

// Fann represents a neural network
type Fann[T Numeric] struct {
	// Error handling
	errno  ErrorCode
	errstr string

	// Network structure
	layers      []layer[T]
	neurons     []neuron[T]
	weights     []T   // All weights in a contiguous array
	connections []int // Indices of source neurons

	// Network topology
	networkType      NetworkType
	connectionRate   float32
	numInput         int
	numOutput        int
	totalNeurons     int
	totalConnections int

	// Training state
	trainErrors     []T   // Error values during training
	trainSlopes     []T   // Slopes for batch training
	prevSteps       []T   // Previous steps for RPROP
	prevTrainSlopes []T   // Previous slopes for RPROP/quickprop
	prevWeightDeltas []T  // Previous weight deltas

	// Training parameters
	trainingAlgorithm   TrainAlgorithm
	learningRate        float32
	learningMomentum    float32
	trainErrorFunction  ErrorFunc
	trainStopFunction   int
	bitFailLimit        T
	
	// RPROP parameters
	rpropIncreaseFactor float32
	rpropDecreaseFactor float32
	rpropDeltaMin      float32
	rpropDeltaMax      float32
	rpropDeltaZero     float32
	
	// Quickprop parameters
	quickpropDecay float32
	quickpropMu    float32
	
	// Sarprop parameters
	sarpropWeightDecayShift         float32
	sarpropStepErrorThresholdFactor float32
	sarpropStepErrorShift           float32
	sarpropTemperature              float32
	sarpropEpoch                    int
	
	// Cascade training parameters
	cascadeOutputChangeFraction      float32
	cascadeOutputStagnationEpochs    int
	cascadeCandidateChangeFraction   float32
	cascadeCandidateStagnationEpochs int
	cascadeCandidateLimit            int
	cascadeMaxOutEpochs              int
	cascadeMaxCandEpochs             int
	cascadeMinOutEpochs              int
	cascadeMinCandEpochs             int
	cascadeCandidates                []neuron[T]
	cascadeCandidateScores           []T
	cascadeNumCandidateGroups        int
	cascadeActivationFunctions       []ActivationFunc
	cascadeActivationSteepnesses     []T
	cascadeWeightMultiplier          T
	
	// MSE calculation
	mse        float32
	numMSE     int
	bitFail    int

	// Callback function
	callback func(ann *Fann[T], epochs int, mse float32) bool
	
	// Scaling parameters
	scaleParams *ScaleParams[T]
}

// CreateStandard creates a standard fully connected neural network
func CreateStandard[T Numeric](layers ...int) *Fann[T] {
	if len(layers) < 2 {
		return nil
	}
	return createNetwork[T](NetTypeLayer, 1.0, layers)
}

// CreateStandardArray creates a standard network from layer array
func CreateStandardArray[T Numeric](layers []int) *Fann[T] {
	if len(layers) < 2 {
		return nil
	}
	return createNetwork[T](NetTypeLayer, 1.0, layers)
}

// CreateSparse creates a sparse connected neural network
func CreateSparse[T Numeric](connectionRate float32, layers ...int) *Fann[T] {
	if len(layers) < 2 || connectionRate <= 0 || connectionRate > 1 {
		return nil
	}
	return createNetwork[T](NetTypeLayer, connectionRate, layers)
}

// CreateSparseArray creates a sparse network from layer array
func CreateSparseArray[T Numeric](connectionRate float32, layers []int) *Fann[T] {
	if len(layers) < 2 || connectionRate <= 0 || connectionRate > 1 {
		return nil
	}
	return createNetwork[T](NetTypeLayer, connectionRate, layers)
}

// CreateShortcut creates a shortcut connected neural network
func CreateShortcut[T Numeric](layers ...int) *Fann[T] {
	if len(layers) < 2 {
		return nil
	}
	return createNetwork[T](NetTypeShortcut, 1.0, layers)
}

// CreateShortcutArray creates a shortcut network from layer array
func CreateShortcutArray[T Numeric](layers []int) *Fann[T] {
	if len(layers) < 2 {
		return nil
	}
	return createNetwork[T](NetTypeShortcut, 1.0, layers)
}

// createNetwork is the internal network creation function
func createNetwork[T Numeric](netType NetworkType, connectionRate float32, layers []int) *Fann[T] {
	ann := &Fann[T]{
		networkType:       netType,
		connectionRate:    connectionRate,
		learningRate:      DefaultLearningRate,
		learningMomentum:  DefaultLearningMomentum,
		trainingAlgorithm: TrainRPROP,
		trainErrorFunction: ErrorTanh,
		bitFailLimit:      T(DefaultBitFailLimit),
		// RPROP defaults
		rpropIncreaseFactor: DefaultRpropIncreaseFactor,
		rpropDecreaseFactor: DefaultRpropDecreaseFactor,
		rpropDeltaMin:      DefaultRpropDeltaMin,
		rpropDeltaMax:      DefaultRpropDeltaMax,
		rpropDeltaZero:     DefaultRpropDeltaZero,
		// Quickprop defaults
		quickpropDecay: DefaultQuickpropDecay,
		quickpropMu:    DefaultQuickpropMu,
		// Sarprop defaults
		sarpropWeightDecayShift:         DefaultSarpropWeightDecayShift,
		sarpropStepErrorThresholdFactor: DefaultSarpropStepErrorThresholdFactor,
		sarpropStepErrorShift:           DefaultSarpropStepErrorShift,
		sarpropTemperature:              DefaultSarpropTemperature,
		sarpropEpoch:                    0,
	}

	ann.numInput = layers[0]
	ann.numOutput = layers[len(layers)-1]

	// Calculate total neurons (including bias neurons)
	totalNeurons := 0
	for i, size := range layers {
		totalNeurons += size
		if i != len(layers)-1 { // Add bias neuron for all but output layer
			totalNeurons++
		}
	}
	ann.totalNeurons = totalNeurons

	// Allocate neurons
	ann.neurons = make([]neuron[T], totalNeurons)
	
	// Set up layers
	ann.layers = make([]layer[T], len(layers))
	neuronIdx := 0
	for i, size := range layers {
		ann.layers[i].firstNeuron = neuronIdx
		neuronIdx += size
		if i != len(layers)-1 { // Add bias neuron
			neuronIdx++
		}
		ann.layers[i].lastNeuron = neuronIdx
	}

	// Initialize neurons
	for i := range ann.neurons {
		ann.neurons[i].activationFunction = Sigmoid
		ann.neurons[i].activationSteepness = T(DefaultActivationSteepnessHidden)
	}

	// Set up connections
	ann.setupConnections(netType, connectionRate)

	// Initialize weights randomly
	ann.RandomizeWeights(-0.1, 0.1)

	return ann
}

// setupConnections creates the connection topology
func (ann *Fann[T]) setupConnections(netType NetworkType, connectionRate float32) {
	totalConnections := 0
	
	// Calculate total connections
	switch netType {
	case NetTypeLayer:
		// Standard layer-to-layer connections
		for i := 0; i < len(ann.layers)-1; i++ {
			numSrcNeurons := ann.layers[i].lastNeuron - ann.layers[i].firstNeuron
			numDstNeurons := ann.layers[i+1].lastNeuron - ann.layers[i+1].firstNeuron
			if i < len(ann.layers)-2 { // Exclude bias neuron from destination count
				numDstNeurons--
			}
			connections := int(float32(numSrcNeurons*numDstNeurons) * connectionRate)
			totalConnections += connections
		}
	case NetTypeShortcut:
		// Each layer connects to all following layers
		for i := 0; i < len(ann.layers)-1; i++ {
			numSrcNeurons := ann.layers[i].lastNeuron - ann.layers[i].firstNeuron
			for j := i + 1; j < len(ann.layers); j++ {
				numDstNeurons := ann.layers[j].lastNeuron - ann.layers[j].firstNeuron
				if j < len(ann.layers)-1 { // Exclude bias neuron
					numDstNeurons--
				}
				connections := int(float32(numSrcNeurons*numDstNeurons) * connectionRate)
				totalConnections += connections
			}
		}
	}

	ann.totalConnections = totalConnections
	ann.weights = make([]T, totalConnections)
	ann.connections = make([]int, totalConnections)

	// Set up connection indices
	connIdx := 0
	switch netType {
	case NetTypeLayer:
		for i := 0; i < len(ann.layers)-1; i++ {
			srcLayer := ann.layers[i]
			dstLayer := ann.layers[i+1]
			
			for dst := dstLayer.firstNeuron; dst < dstLayer.lastNeuron; dst++ {
				if i < len(ann.layers)-2 && dst == dstLayer.lastNeuron-1 {
					continue // Skip bias neuron connections
				}
				
				ann.neurons[dst].firstCon = connIdx
				
				// Connect to all neurons in previous layer (including bias)
				for src := srcLayer.firstNeuron; src < srcLayer.lastNeuron; src++ {
					if connectionRate < 1.0 && (connIdx-ann.neurons[dst].firstCon) >= int(float32(srcLayer.lastNeuron-srcLayer.firstNeuron)*connectionRate) {
						break
					}
					ann.connections[connIdx] = src
					connIdx++
				}
				
				ann.neurons[dst].lastCon = connIdx
			}
		}
	case NetTypeShortcut:
		// Each layer connects to ALL following layers (not just the next one)
		for i := 0; i < len(ann.layers)-1; i++ {
			srcLayer := ann.layers[i]
			
			// Connect to all layers that come after this one
			for j := i + 1; j < len(ann.layers); j++ {
				dstLayer := ann.layers[j]
				
				for dst := dstLayer.firstNeuron; dst < dstLayer.lastNeuron; dst++ {
					// Skip bias neuron connections (bias neurons don't receive inputs)
					if j < len(ann.layers)-1 && dst == dstLayer.lastNeuron-1 {
						continue
					}
					
					// Set connection range for this destination neuron
					if ann.neurons[dst].firstCon == 0 && connIdx > 0 {
						ann.neurons[dst].firstCon = connIdx
					}
					
					// Connect to all neurons in this source layer
					for src := srcLayer.firstNeuron; src < srcLayer.lastNeuron; src++ {
						if connectionRate < 1.0 {
							// For sparse shortcut networks, limit connections per neuron
							currentConnections := connIdx - ann.neurons[dst].firstCon
							maxConnections := int(float32(srcLayer.lastNeuron-srcLayer.firstNeuron) * connectionRate)
							if currentConnections >= maxConnections {
								break
							}
						}
						ann.connections[connIdx] = src
						connIdx++
					}
					
					ann.neurons[dst].lastCon = connIdx
				}
			}
		}
	}
}

// RandomizeWeights gives each connection a random weight between minWeight and maxWeight
func (ann *Fann[T]) RandomizeWeights(minWeight, maxWeight T) {
	for i := range ann.weights {
		// Generate random float between 0 and 1
		randFloat := float64(randomUint64()&0x1FFFFFFFFFFFFF) / float64(1<<53)
		ann.weights[i] = minWeight + T(randFloat)*(maxWeight-minWeight)
	}
}

// Run executes the neural network with the given input
func (ann *Fann[T]) Run(input []T) []T {
	if len(input) != ann.numInput {
		ann.setError(ErrInputMismatch, fmt.Sprintf("expected %d inputs, got %d", ann.numInput, len(input)))
		return nil
	}

	// Set input values
	inputLayer := ann.layers[0]
	for i := 0; i < ann.numInput; i++ {
		ann.neurons[inputLayer.firstNeuron+i].value = input[i]
	}

	// Set bias neurons to 1
	for i := 0; i < len(ann.layers)-1; i++ {
		biasIdx := ann.layers[i].lastNeuron - 1
		ann.neurons[biasIdx].value = T(1.0)
	}

	// Forward propagation
	for i := 1; i < len(ann.layers); i++ {
		layer := ann.layers[i]
		
		for j := layer.firstNeuron; j < layer.lastNeuron; j++ {
			if i < len(ann.layers)-1 && j == layer.lastNeuron-1 {
				continue // Skip bias neuron
			}
			
			neuron := &ann.neurons[j]
			
			// Calculate weighted sum
			sum := T(0)
			for k := neuron.firstCon; k < neuron.lastCon; k++ {
				sum += ann.weights[k] * ann.neurons[ann.connections[k]].value
			}
			neuron.sum = sum
			
			// Apply activation function
			neuron.value = ann.activation(neuron.activationFunction, neuron.activationSteepness, sum)
		}
	}

	// Extract output
	outputLayer := ann.layers[len(ann.layers)-1]
	output := make([]T, ann.numOutput)
	for i := 0; i < ann.numOutput; i++ {
		output[i] = ann.neurons[outputLayer.firstNeuron+i].value
	}

	return output
}

// activation applies the activation function
func (ann *Fann[T]) activation(fn ActivationFunc, steepness, value T) T {
	return Activation(fn, steepness, value)
}

// Helper functions

func abs[T Numeric](x T) T {
	if x < 0 {
		return -x
	}
	return x
}

// setError sets the error state
func (ann *Fann[T]) setError(code ErrorCode, message string) {
	ann.errno = code
	ann.errstr = message
}

// GetError returns the current error code
func (ann *Fann[T]) GetError() ErrorCode {
	return ann.errno
}

// GetErrorString returns the current error message
func (ann *Fann[T]) GetErrorString() string {
	return ann.errstr
}

// ResetError clears the error state
func (ann *Fann[T]) ResetError() {
	ann.errno = ErrNoError
	ann.errstr = ""
}

// Connection represents a connection between neurons
type Connection[T Numeric] struct {
	FromNeuron int // Index of source neuron
	ToNeuron   int // Index of destination neuron
	Weight     T   // Connection weight
}

// Basic getters

// GetNumInput returns the number of input neurons
func (ann *Fann[T]) GetNumInput() int {
	return ann.numInput
}

// GetNumOutput returns the number of output neurons
func (ann *Fann[T]) GetNumOutput() int {
	return ann.numOutput
}

// GetTotalNeurons returns the total number of neurons
func (ann *Fann[T]) GetTotalNeurons() int {
	return ann.totalNeurons
}

// GetTotalConnections returns the total number of connections
func (ann *Fann[T]) GetTotalConnections() int {
	return ann.totalConnections
}

// GetNetworkType returns the network topology type
func (ann *Fann[T]) GetNetworkType() NetworkType {
	return ann.networkType
}

// GetConnectionRate returns the connection rate
func (ann *Fann[T]) GetConnectionRate() float32 {
	return ann.connectionRate
}

// GetConnectionArray returns all connections in the network
func (ann *Fann[T]) GetConnectionArray() []Connection[T] {
	connections := make([]Connection[T], 0, ann.totalConnections)
	
	for toNeuron := 0; toNeuron < ann.totalNeurons; toNeuron++ {
		neuron := &ann.neurons[toNeuron]
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			connections = append(connections, Connection[T]{
				FromNeuron: ann.connections[i],
				ToNeuron:   toNeuron,
				Weight:     ann.weights[i],
			})
		}
	}
	
	return connections
}

// SetWeightArray sets weights for existing connections
func (ann *Fann[T]) SetWeightArray(connections []Connection[T]) error {
	// Create a map for quick lookup
	connectionMap := make(map[struct{ from, to int }]T)
	for _, conn := range connections {
		connectionMap[struct{ from, to int }{conn.FromNeuron, conn.ToNeuron}] = conn.Weight
	}
	
	// Update existing connections
	modified := 0
	for toNeuron := 0; toNeuron < ann.totalNeurons; toNeuron++ {
		neuron := &ann.neurons[toNeuron]
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			fromNeuron := ann.connections[i]
			if weight, exists := connectionMap[struct{ from, to int }{fromNeuron, toNeuron}]; exists {
				ann.weights[i] = weight
				modified++
			}
		}
	}
	
	if modified == 0 {
		return fmt.Errorf("no connections were modified")
	}
	
	return nil
}

// GetNumLayers returns the number of layers in the network
func (ann *Fann[T]) GetNumLayers() int {
	return len(ann.layers)
}

// GetLayerArray returns the number of neurons in each layer
func (ann *Fann[T]) GetLayerArray() []int {
	layers := make([]int, len(ann.layers))
	for i, layer := range ann.layers {
		numNeurons := layer.lastNeuron - layer.firstNeuron
		if i < len(ann.layers)-1 {
			numNeurons-- // Exclude bias neuron
		}
		layers[i] = numNeurons
	}
	return layers
}

// GetBiasArray returns the bias values for each layer
func (ann *Fann[T]) GetBiasArray() []T {
	biases := make([]T, 0, len(ann.layers)-1)
	for i := 0; i < len(ann.layers)-1; i++ {
		biasIdx := ann.layers[i].lastNeuron - 1
		biases = append(biases, ann.neurons[biasIdx].value)
	}
	return biases
}

// GetWeight returns the weight of a connection
func (ann *Fann[T]) GetWeight(fromNeuron, toNeuron int) (T, error) {
	// Validate neuron indices
	if fromNeuron < 0 || fromNeuron >= ann.totalNeurons {
		return 0, fmt.Errorf("invalid from_neuron index: %d", fromNeuron)
	}
	if toNeuron < 0 || toNeuron >= ann.totalNeurons {
		return 0, fmt.Errorf("invalid to_neuron index: %d", toNeuron)
	}
	
	// Find the connection
	neuron := &ann.neurons[toNeuron]
	for i := neuron.firstCon; i < neuron.lastCon; i++ {
		if ann.connections[i] == fromNeuron {
			return ann.weights[i], nil
		}
	}
	
	return 0, fmt.Errorf("no connection from neuron %d to neuron %d", fromNeuron, toNeuron)
}

// SetWeight sets the weight of a connection
func (ann *Fann[T]) SetWeight(fromNeuron, toNeuron int, weight T) error {
	// Validate neuron indices
	if fromNeuron < 0 || fromNeuron >= ann.totalNeurons {
		return fmt.Errorf("invalid from_neuron index: %d", fromNeuron)
	}
	if toNeuron < 0 || toNeuron >= ann.totalNeurons {
		return fmt.Errorf("invalid to_neuron index: %d", toNeuron)
	}
	
	// Find the connection
	neuron := &ann.neurons[toNeuron]
	for i := neuron.firstCon; i < neuron.lastCon; i++ {
		if ann.connections[i] == fromNeuron {
			ann.weights[i] = weight
			return nil
		}
	}
	
	return fmt.Errorf("no connection from neuron %d to neuron %d", fromNeuron, toNeuron)
}

// GetWeights returns all weights as a slice
func (ann *Fann[T]) GetWeights() []T {
	result := make([]T, len(ann.weights))
	copy(result, ann.weights)
	return result
}

// SetWeights sets all weights from a slice
func (ann *Fann[T]) SetWeights(weights []T) error {
	if len(weights) != len(ann.weights) {
		return fmt.Errorf("weight count mismatch: expected %d, got %d", len(ann.weights), len(weights))
	}
	copy(ann.weights, weights)
	return nil
}

// CreateCascade creates a minimal network for cascade training
func CreateCascade[T Numeric](numInput, numOutput int) *Fann[T] {
	// Create network with no hidden layers
	net := CreateStandard[T](numInput, numOutput)
	
	// Set default cascade parameters
	net.cascadeOutputChangeFraction = 0.01
	net.cascadeOutputStagnationEpochs = 12
	net.cascadeCandidateChangeFraction = 0.01
	net.cascadeCandidateStagnationEpochs = 12
	net.cascadeMaxOutEpochs = 150
	net.cascadeMaxCandEpochs = 150
	net.cascadeMinOutEpochs = 50
	net.cascadeMinCandEpochs = 50
	net.cascadeCandidateLimit = 1000
	net.cascadeNumCandidateGroups = 2
	
	// Set default activation functions for cascade
	net.cascadeActivationFunctions = []ActivationFunc{
		Sigmoid, SigmoidSymmetric, Gaussian, GaussianSymmetric,
		Elliot, ElliotSymmetric, SinSymmetric, CosSymmetric,
		Sin, Cos,
	}
	
	// Set default steepnesses
	net.cascadeActivationSteepnesses = []T{T(0.25), T(0.5), T(0.75), T(1.0)}
	
	// Set cascade weight multiplier
	net.cascadeWeightMultiplier = T(DefaultCascadeWeightMultiplier)
	
	// Set more conservative RPROP parameters for cascade
	net.rpropDeltaZero = 0.01      // Smaller initial step
	net.rpropDeltaMax = 5.0        // Smaller max step
	
	// Use Batch training by default for cascade (RPROP can cause output collapse)
	net.SetTrainingAlgorithm(TrainBatch)
	net.SetLearningRate(DefaultLearningRate) // Higher learning rate for cascade
	
	return net
}

// SetActivationFunction sets the activation function for a specific neuron
func (ann *Fann[T]) SetActivationFunction(activation ActivationFunc, layer, neuron int) error {
	if layer < 0 || layer >= len(ann.layers) {
		return fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return fmt.Errorf("cannot set activation function for input layer")
	}
	
	layerStruct := ann.layers[layer]
	numNeurons := layerStruct.lastNeuron - layerStruct.firstNeuron
	if layer < len(ann.layers)-1 {
		numNeurons-- // Exclude bias neuron
	}
	
	if neuron < 0 || neuron >= numNeurons {
		return fmt.Errorf("invalid neuron index: %d", neuron)
	}
	
	neuronIdx := layerStruct.firstNeuron + neuron
	ann.neurons[neuronIdx].activationFunction = activation
	return nil
}

// GetActivationFunction gets the activation function for a specific neuron
func (ann *Fann[T]) GetActivationFunction(layer, neuron int) (ActivationFunc, error) {
	if layer < 0 || layer >= len(ann.layers) {
		return 0, fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return 0, fmt.Errorf("cannot get activation function for input layer")
	}
	
	layerStruct := ann.layers[layer]
	numNeurons := layerStruct.lastNeuron - layerStruct.firstNeuron
	if layer < len(ann.layers)-1 {
		numNeurons-- // Exclude bias neuron
	}
	
	if neuron < 0 || neuron >= numNeurons {
		return 0, fmt.Errorf("invalid neuron index: %d", neuron)
	}
	
	neuronIdx := layerStruct.firstNeuron + neuron
	return ann.neurons[neuronIdx].activationFunction, nil
}

// SetActivationFunctionLayer sets the activation function for all neurons in a layer
func (ann *Fann[T]) SetActivationFunctionLayer(activation ActivationFunc, layer int) error {
	if layer < 0 || layer >= len(ann.layers) {
		return fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return fmt.Errorf("cannot set activation function for input layer")
	}
	
	layerStruct := ann.layers[layer]
	for i := layerStruct.firstNeuron; i < layerStruct.lastNeuron; i++ {
		if layer < len(ann.layers)-1 && i == layerStruct.lastNeuron-1 {
			continue // Skip bias neuron
		}
		ann.neurons[i].activationFunction = activation
	}
	return nil
}

// SetActivationFunctionHidden sets the activation function for all hidden layers
func (ann *Fann[T]) SetActivationFunctionHidden(activation ActivationFunc) error {
	for i := 1; i < len(ann.layers)-1; i++ {
		if err := ann.SetActivationFunctionLayer(activation, i); err != nil {
			return err
		}
	}
	return nil
}

// SetActivationFunctionOutput sets the activation function for the output layer
func (ann *Fann[T]) SetActivationFunctionOutput(activation ActivationFunc) error {
	return ann.SetActivationFunctionLayer(activation, len(ann.layers)-1)
}

// SetActivationSteepness sets the activation steepness for a specific neuron
func (ann *Fann[T]) SetActivationSteepness(steepness T, layer, neuron int) error {
	if layer < 0 || layer >= len(ann.layers) {
		return fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return fmt.Errorf("cannot set activation steepness for input layer")
	}
	
	layerStruct := ann.layers[layer]
	numNeurons := layerStruct.lastNeuron - layerStruct.firstNeuron
	if layer < len(ann.layers)-1 {
		numNeurons-- // Exclude bias neuron
	}
	
	if neuron < 0 || neuron >= numNeurons {
		return fmt.Errorf("invalid neuron index: %d", neuron)
	}
	
	neuronIdx := layerStruct.firstNeuron + neuron
	ann.neurons[neuronIdx].activationSteepness = steepness
	return nil
}

// GetActivationSteepness gets the activation steepness for a specific neuron
func (ann *Fann[T]) GetActivationSteepness(layer, neuron int) (T, error) {
	if layer < 0 || layer >= len(ann.layers) {
		return 0, fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return 0, fmt.Errorf("cannot get activation steepness for input layer")
	}
	
	layerStruct := ann.layers[layer]
	numNeurons := layerStruct.lastNeuron - layerStruct.firstNeuron
	if layer < len(ann.layers)-1 {
		numNeurons-- // Exclude bias neuron
	}
	
	if neuron < 0 || neuron >= numNeurons {
		return 0, fmt.Errorf("invalid neuron index: %d", neuron)
	}
	
	neuronIdx := layerStruct.firstNeuron + neuron
	return ann.neurons[neuronIdx].activationSteepness, nil
}

// SetActivationSteepnessLayer sets the activation steepness for all neurons in a layer
func (ann *Fann[T]) SetActivationSteepnessLayer(steepness T, layer int) error {
	if layer < 0 || layer >= len(ann.layers) {
		return fmt.Errorf("invalid layer index: %d", layer)
	}
	if layer == 0 {
		return fmt.Errorf("cannot set activation steepness for input layer")
	}
	
	layerStruct := ann.layers[layer]
	for i := layerStruct.firstNeuron; i < layerStruct.lastNeuron; i++ {
		if layer < len(ann.layers)-1 && i == layerStruct.lastNeuron-1 {
			continue // Skip bias neuron
		}
		ann.neurons[i].activationSteepness = steepness
	}
	return nil
}

// SetActivationSteepnessHidden sets the activation steepness for all hidden layers
func (ann *Fann[T]) SetActivationSteepnessHidden(steepness T) error {
	for i := 1; i < len(ann.layers)-1; i++ {
		if err := ann.SetActivationSteepnessLayer(steepness, i); err != nil {
			return err
		}
	}
	return nil
}

// SetActivationSteepnessOutput sets the activation steepness for the output layer
func (ann *Fann[T]) SetActivationSteepnessOutput(steepness T) error {
	return ann.SetActivationSteepnessLayer(steepness, len(ann.layers)-1)
}

// Simple random number generator for weight initialization
var randState uint64 = 1

func randomUint64() uint64 {
	randState = randState*1664525 + 1013904223
	return randState
}