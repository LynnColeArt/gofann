package gofann

import (
	"fmt"
	"math"
	"runtime"
)


// Train trains the network with one input-output pair (online/incremental training)
func (ann *Fann[T]) Train(input, desiredOutput []T) {
	if len(input) != ann.numInput {
		ann.setError(ErrInputMismatch, "training input size mismatch")
		return
	}
	if len(desiredOutput) != ann.numOutput {
		ann.setError(ErrOutputMismatch, "training output size mismatch")
		return
	}

	// Run forward pass
	output := ann.Run(input)
	if output == nil {
		return // Error already set
	}

	// Calculate output layer errors
	outputLayer := ann.layers[len(ann.layers)-1]
	
	// Allocate training arrays if needed
	if ann.trainErrors == nil {
		ann.trainErrors = make([]T, ann.totalNeurons)
		ann.trainSlopes = make([]T, ann.totalConnections)
	}

	// Calculate errors for output layer
	for i := 0; i < ann.numOutput; i++ {
		neuronIdx := outputLayer.firstNeuron + i
		neuron := &ann.neurons[neuronIdx]
		
		// Calculate error
		error := desiredOutput[i] - neuron.value
		
		// Calculate error derivative
		ann.trainErrors[neuronIdx] = error * ActivationDerivative(
			neuron.activationFunction,
			neuron.activationSteepness,
			neuron.value,
			neuron.sum,
		)
	}

	// Backpropagate errors
	for layerIdx := len(ann.layers) - 2; layerIdx >= 0; layerIdx-- {
		layer := ann.layers[layerIdx]
		nextLayer := ann.layers[layerIdx+1]
		
		// Clear errors for this layer
		for i := layer.firstNeuron; i < layer.lastNeuron; i++ {
			ann.trainErrors[i] = 0
		}
		
		// Accumulate errors from next layer
		for i := nextLayer.firstNeuron; i < nextLayer.lastNeuron; i++ {
			if layerIdx < len(ann.layers)-2 && i == nextLayer.lastNeuron-1 {
				continue // Skip bias neuron
			}
			
			neuron := &ann.neurons[i]
			error := ann.trainErrors[i]
			
			// Propagate error to previous layer
			for j := neuron.firstCon; j < neuron.lastCon; j++ {
				ann.trainErrors[ann.connections[j]] += error * ann.weights[j]
			}
		}
		
		// Apply activation derivative
		for i := layer.firstNeuron; i < layer.lastNeuron; i++ {
			if layerIdx > 0 && i == layer.lastNeuron-1 {
				continue // Skip bias neuron
			}
			
			neuron := &ann.neurons[i]
			ann.trainErrors[i] *= ActivationDerivative(
				neuron.activationFunction,
				neuron.activationSteepness,
				neuron.value,
				neuron.sum,
			)
		}
	}

	// Update weights based on training algorithm
	switch ann.trainingAlgorithm {
	case TrainIncremental:
		ann.updateWeightsIncremental()
	case TrainBatch:
		ann.updateSlopesBatch()
	case TrainRPROP:
		ann.updateSlopesBatch()
		// RPROP weight update happens after epoch
	case TrainQuickprop:
		ann.updateSlopesBatch()
		// Quickprop weight update happens after epoch
	case TrainSarprop:
		ann.updateSlopesBatch()
		// Sarprop weight update happens after epoch
	default:
		ann.updateWeightsIncremental()
	}
	
	// Update MSE
	ann.calculateMSE(desiredOutput)
}

// updateWeightsIncremental updates weights immediately (online learning)
func (ann *Fann[T]) updateWeightsIncremental() {
	learningRate := T(ann.learningRate)
	momentum := T(ann.learningMomentum)
	
	// Allocate momentum array if needed
	if momentum != 0 && ann.prevWeightDeltas == nil {
		ann.prevWeightDeltas = make([]T, ann.totalConnections)
	}
	
	// Update all weights
	connIdx := 0
	for layerIdx := 1; layerIdx < len(ann.layers); layerIdx++ {
		layer := ann.layers[layerIdx]
		
		for neuronIdx := layer.firstNeuron; neuronIdx < layer.lastNeuron; neuronIdx++ {
			if layerIdx < len(ann.layers)-1 && neuronIdx == layer.lastNeuron-1 {
				continue // Skip bias neuron
			}
			
			neuron := &ann.neurons[neuronIdx]
			error := ann.trainErrors[neuronIdx]
			
			for i := neuron.firstCon; i < neuron.lastCon; i++ {
				// Calculate weight delta
				delta := learningRate * error * ann.neurons[ann.connections[i]].value
				
				// Add momentum if enabled
				if momentum != 0 {
					delta += momentum * ann.prevWeightDeltas[i]
					ann.prevWeightDeltas[i] = delta
				}
				
				// Update weight
				ann.weights[i] += delta
				connIdx++
			}
		}
	}
}

// updateSlopesBatch accumulates slopes for batch training
func (ann *Fann[T]) updateSlopesBatch() {
	// Update slopes for all connections
	connIdx := 0
	for layerIdx := 1; layerIdx < len(ann.layers); layerIdx++ {
		layer := ann.layers[layerIdx]
		
		for neuronIdx := layer.firstNeuron; neuronIdx < layer.lastNeuron; neuronIdx++ {
			if layerIdx < len(ann.layers)-1 && neuronIdx == layer.lastNeuron-1 {
				continue // Skip bias neuron
			}
			
			neuron := &ann.neurons[neuronIdx]
			error := ann.trainErrors[neuronIdx]
			
			for i := neuron.firstCon; i < neuron.lastCon; i++ {
				ann.trainSlopes[i] += error * ann.neurons[ann.connections[i]].value
				connIdx++
			}
		}
	}
}

// TrainEpoch trains for one epoch and returns MSE
func (ann *Fann[T]) TrainEpoch(data *TrainData[T]) float32 {
	ann.ResetMSE()
	
	// Clear slopes for batch training
	if ann.trainingAlgorithm != TrainIncremental {
		for i := range ann.trainSlopes {
			ann.trainSlopes[i] = 0
		}
	}
	
	// Train on each pattern
	for i := 0; i < data.numData; i++ {
		ann.Train(data.inputs[i], data.outputs[i])
		
		// Yield CPU occasionally on large datasets
		if data.numData > 1000 && i%1000 == 0 {
			runtime.Gosched()
		}
	}
	
	// Update weights for batch algorithms
	switch ann.trainingAlgorithm {
	case TrainBatch:
		ann.updateWeightsBatch(data.numData)
	case TrainRPROP:
		ann.updateWeightsRPROP(data.numData)
	case TrainQuickprop:
		ann.updateWeightsQuickprop(data.numData)
	case TrainSarprop:
		ann.updateWeightsSarprop(data.numData)
	}
	
	return ann.GetMSE()
}

// updateWeightsBatch updates weights after accumulating slopes
func (ann *Fann[T]) updateWeightsBatch(numData int) {
	learningRate := T(ann.learningRate) / T(numData)
	momentum := T(ann.learningMomentum)
	
	// Allocate momentum array if needed
	if momentum != 0 && ann.prevWeightDeltas == nil {
		ann.prevWeightDeltas = make([]T, ann.totalConnections)
	}
	
	// Update all weights
	for i := 0; i < ann.totalConnections; i++ {
		// Calculate weight delta (negative gradient)
		delta := -learningRate * ann.trainSlopes[i]
		
		// Add momentum if enabled
		if momentum != 0 {
			delta += momentum * ann.prevWeightDeltas[i]
			ann.prevWeightDeltas[i] = delta
		}
		
		// Update weight
		ann.weights[i] += delta
	}
}

// updateWeightsRPROP implements the iRPROP- algorithm
func (ann *Fann[T]) updateWeightsRPROP(numData int) {
	// Use configured RPROP parameters
	increaseFactor := ann.rpropIncreaseFactor
	decreaseFactor := ann.rpropDecreaseFactor
	deltaMin := ann.rpropDeltaMin
	deltaMax := ann.rpropDeltaMax
	
	// Initialize RPROP arrays if needed
	if ann.prevSteps == nil {
		ann.prevSteps = make([]T, ann.totalConnections)
		ann.prevTrainSlopes = make([]T, ann.totalConnections)
		
		// Initialize steps
		for i := range ann.prevSteps {
			ann.prevSteps[i] = T(ann.rpropDeltaZero) // Initial step size
		}
	}
	
	// Update each weight
	for i := 0; i < ann.totalConnections; i++ {
		prevSlope := ann.prevTrainSlopes[i]
		slope := ann.trainSlopes[i] / T(numData) // Normalize by number of samples
		prevStep := ann.prevSteps[i]
		
		// Calculate sign change
		signChange := prevSlope * slope
		
		if signChange >= 0 {
			// No sign change or same direction
			step := prevStep * T(increaseFactor)
			if step > T(deltaMax) {
				step = T(deltaMax)
			}
			ann.prevSteps[i] = step
			
			// Update weight
			if slope > 0 {
				ann.weights[i] -= step
			} else if slope < 0 {
				ann.weights[i] += step
			}
			
			ann.prevTrainSlopes[i] = slope
		} else {
			// Sign change - reduce step size
			step := prevStep * T(decreaseFactor)
			if step < T(deltaMin) {
				step = T(deltaMin)
			}
			ann.prevSteps[i] = step
			
			// Don't update weight on sign change (iRPROP-)
			ann.prevTrainSlopes[i] = 0
		}
	}
}

// calculateMSE updates the MSE calculation
func (ann *Fann[T]) calculateMSE(desiredOutput []T) {
	outputLayer := ann.layers[len(ann.layers)-1]
	
	mse := float32(0)
	for i := 0; i < ann.numOutput; i++ {
		neuronIdx := outputLayer.firstNeuron + i
		diff := float32(desiredOutput[i] - ann.neurons[neuronIdx].value)
		mse += diff * diff
		
		// Count bit failures
		if math.Abs(float64(diff)) > float64(ann.bitFailLimit) {
			ann.bitFail++
		}
	}
	
	ann.mse += mse
	ann.numMSE++
}

// GetMSE returns the mean squared error
func (ann *Fann[T]) GetMSE() float32 {
	if ann.numMSE == 0 || ann.numOutput == 0 {
		return 0
	}
	return ann.mse / float32(ann.numMSE) / float32(ann.numOutput)
}

// ResetMSE resets the MSE calculation
func (ann *Fann[T]) ResetMSE() {
	ann.mse = 0
	ann.numMSE = 0
	ann.bitFail = 0
}

// GetBitFail returns the number of fail bits
func (ann *Fann[T]) GetBitFail() int {
	return ann.bitFail
}

// Training parameter setters

// SetTrainingAlgorithm sets the training algorithm
func (ann *Fann[T]) SetTrainingAlgorithm(algorithm TrainAlgorithm) {
	ann.trainingAlgorithm = algorithm
}

// GetTrainingAlgorithm returns the current training algorithm
func (ann *Fann[T]) GetTrainingAlgorithm() TrainAlgorithm {
	return ann.trainingAlgorithm
}

// SetLearningRate sets the learning rate
func (ann *Fann[T]) SetLearningRate(rate float32) {
	ann.learningRate = rate
}

// GetLearningRate returns the learning rate
func (ann *Fann[T]) GetLearningRate() float32 {
	return ann.learningRate
}

// SetLearningMomentum sets the learning momentum
func (ann *Fann[T]) SetLearningMomentum(momentum float32) {
	ann.learningMomentum = momentum
}

// GetLearningMomentum returns the learning momentum
func (ann *Fann[T]) GetLearningMomentum() float32 {
	return ann.learningMomentum
}

// updateWeightsQuickprop implements the Quickprop algorithm
func (ann *Fann[T]) updateWeightsQuickprop(numData int) {
	// Quickprop parameters
	decay := T(ann.quickpropDecay)
	mu := T(ann.quickpropMu)
	epsilon := T(ann.learningRate) / T(numData)
	
	// Initialize Quickprop arrays if needed
	if ann.prevTrainSlopes == nil {
		ann.prevTrainSlopes = make([]T, ann.totalConnections)
		ann.prevWeightDeltas = make([]T, ann.totalConnections)
	}
	
	// Update each weight
	for i := 0; i < ann.totalConnections; i++ {
		w := ann.weights[i]
		prevSlope := ann.prevTrainSlopes[i]
		slope := ann.trainSlopes[i] / T(numData) + decay * w
		prevDelta := ann.prevWeightDeltas[i]
		
		delta := T(0)
		
		// Check if this is the first iteration
		if prevDelta == 0 && prevSlope == 0 {
			// First iteration - use simple gradient descent
			delta = -epsilon * slope
		} else {
			// Subsequent iterations - use Quickprop
			slopeDiff := slope - prevSlope
			
			// Standard Quickprop formula
			if abs(slopeDiff) > T(EpsilonQuickprop) && abs(prevDelta) > T(EpsilonQuickprop) {
				// Calculate Quickprop step
				delta = -slope * prevDelta / slopeDiff
				
				// Limit step size
				maxDelta := mu * abs(prevDelta)
				if abs(delta) > maxDelta {
					if delta > 0 {
						delta = maxDelta
					} else {
						delta = -maxDelta
					}
				}
				
				// Add gradient descent term if slopes have same sign
				if prevSlope * slope > 0 {
					delta += epsilon * slope
				}
			} else {
				// Fall back to gradient descent
				delta = -epsilon * slope
			}
		}
		
		// Apply the weight change
		ann.weights[i] += delta
		
		// Save for next iteration
		ann.prevTrainSlopes[i] = slope
		ann.prevWeightDeltas[i] = delta
	}
}

// Quickprop parameter setters/getters

// SetQuickpropDecay sets the decay factor for Quickprop
func (ann *Fann[T]) SetQuickpropDecay(decay float32) {
	ann.quickpropDecay = decay
}

// GetQuickpropDecay returns the decay factor for Quickprop
func (ann *Fann[T]) GetQuickpropDecay() float32 {
	return ann.quickpropDecay
}

// SetQuickpropMu sets the mu factor for Quickprop
func (ann *Fann[T]) SetQuickpropMu(mu float32) {
	ann.quickpropMu = mu
}

// GetQuickpropMu returns the mu factor for Quickprop
func (ann *Fann[T]) GetQuickpropMu() float32 {
	return ann.quickpropMu
}

// updateWeightsSarprop implements the SARPROP algorithm
func (ann *Fann[T]) updateWeightsSarprop(numData int) {
	// Sarprop parameters
	decayShift := ann.sarpropWeightDecayShift
	stepErrorThresholdFactor := ann.sarpropStepErrorThresholdFactor
	temperature := ann.sarpropTemperature
	
	// RPROP parameters
	increaseFactor := ann.rpropIncreaseFactor
	decreaseFactor := ann.rpropDecreaseFactor
	deltaMin := ann.rpropDeltaMin
	deltaMax := ann.rpropDeltaMax
	
	// Initialize arrays if needed
	if ann.prevSteps == nil {
		ann.prevSteps = make([]T, ann.totalConnections)
		ann.prevTrainSlopes = make([]T, ann.totalConnections)
		
		// Initialize steps
		for i := range ann.prevSteps {
			ann.prevSteps[i] = T(ann.rpropDeltaZero)
		}
	}
	
	// Calculate MSE for this epoch
	mse := ann.GetMSE()
	
	// Update epoch counter
	ann.sarpropEpoch++
	
	// Update each weight
	for i := 0; i < ann.totalConnections; i++ {
		prevSlope := ann.prevTrainSlopes[i]
		slope := ann.trainSlopes[i] / T(numData)
		prevStep := ann.prevSteps[i]
		weight := ann.weights[i]
		
		// Apply weight decay
		if decayShift < 0 {
			slope += T(math.Pow(2.0, float64(decayShift))) * weight
		} else {
			slope += T(decayShift) * weight
		}
		
		// Calculate step size modification
		signChange := prevSlope * slope
		step := prevStep
		
		if signChange > 0 {
			// Same sign - increase step
			step = prevStep * T(increaseFactor)
			if step > T(deltaMax) {
				step = T(deltaMax)
			}
		} else if signChange < 0 {
			// Different sign - decrease step
			step = prevStep * T(decreaseFactor)
			if step < T(deltaMin) {
				step = T(deltaMin)
			}
			
			// SARPROP: adjust based on temperature and error
			if float32(mse) > stepErrorThresholdFactor * float32(ann.bitFailLimit) {
				// High error - use temperature scaling
				scaleFactor := T(1.0) + T(temperature) * T(ann.sarpropEpoch)
				if scaleFactor > T(1.5) {
					scaleFactor = T(1.5)
				}
				step *= scaleFactor
			}
			
			// Reset slope to avoid double punishment
			slope = 0
		}
		
		// Update weight
		if slope > 0 {
			ann.weights[i] -= step
		} else if slope < 0 {
			ann.weights[i] += step
		}
		
		// Save values for next iteration
		ann.prevSteps[i] = step
		ann.prevTrainSlopes[i] = slope
	}
}

// Sarprop parameter setters/getters

// SetSarpropWeightDecayShift sets the weight decay shift
func (ann *Fann[T]) SetSarpropWeightDecayShift(shift float32) {
	ann.sarpropWeightDecayShift = shift
}

// GetSarpropWeightDecayShift returns the weight decay shift
func (ann *Fann[T]) GetSarpropWeightDecayShift() float32 {
	return ann.sarpropWeightDecayShift
}

// SetSarpropStepErrorThresholdFactor sets the step error threshold factor
func (ann *Fann[T]) SetSarpropStepErrorThresholdFactor(factor float32) {
	ann.sarpropStepErrorThresholdFactor = factor
}

// GetSarpropStepErrorThresholdFactor returns the step error threshold factor
func (ann *Fann[T]) GetSarpropStepErrorThresholdFactor() float32 {
	return ann.sarpropStepErrorThresholdFactor
}

// SetSarpropStepErrorShift sets the step error shift
func (ann *Fann[T]) SetSarpropStepErrorShift(shift float32) {
	ann.sarpropStepErrorShift = shift
}

// GetSarpropStepErrorShift returns the step error shift
func (ann *Fann[T]) GetSarpropStepErrorShift() float32 {
	return ann.sarpropStepErrorShift
}

// SetSarpropTemperature sets the temperature parameter
func (ann *Fann[T]) SetSarpropTemperature(temp float32) {
	ann.sarpropTemperature = temp
}

// GetSarpropTemperature returns the temperature parameter
func (ann *Fann[T]) GetSarpropTemperature() float32 {
	return ann.sarpropTemperature
}

// RPROP parameter setters/getters

// SetRpropIncreaseFactor sets the RPROP increase factor
func (ann *Fann[T]) SetRpropIncreaseFactor(factor float32) {
	ann.rpropIncreaseFactor = factor
}

// GetRpropIncreaseFactor returns the RPROP increase factor
func (ann *Fann[T]) GetRpropIncreaseFactor() float32 {
	return ann.rpropIncreaseFactor
}

// SetRpropDecreaseFactor sets the RPROP decrease factor
func (ann *Fann[T]) SetRpropDecreaseFactor(factor float32) {
	ann.rpropDecreaseFactor = factor
}

// GetRpropDecreaseFactor returns the RPROP decrease factor
func (ann *Fann[T]) GetRpropDecreaseFactor() float32 {
	return ann.rpropDecreaseFactor
}

// SetRpropDeltaMin sets the RPROP minimum step size
func (ann *Fann[T]) SetRpropDeltaMin(delta float32) {
	ann.rpropDeltaMin = delta
}

// GetRpropDeltaMin returns the RPROP minimum step size
func (ann *Fann[T]) GetRpropDeltaMin() float32 {
	return ann.rpropDeltaMin
}

// SetRpropDeltaMax sets the RPROP maximum step size
func (ann *Fann[T]) SetRpropDeltaMax(delta float32) {
	ann.rpropDeltaMax = delta
}

// GetRpropDeltaMax returns the RPROP maximum step size
func (ann *Fann[T]) GetRpropDeltaMax() float32 {
	return ann.rpropDeltaMax
}

// SetRpropDeltaZero sets the RPROP initial step size
func (ann *Fann[T]) SetRpropDeltaZero(delta float32) {
	ann.rpropDeltaZero = delta
}

// GetRpropDeltaZero returns the RPROP initial step size
func (ann *Fann[T]) GetRpropDeltaZero() float32 {
	return ann.rpropDeltaZero
}

// CascadetrainOnData trains using cascade training
func (ann *Fann[T]) CascadetrainOnData(data *TrainData[T], maxNeurons int, 
	neuronsPerCascade int, desiredError float32) {
	
	// Validate inputs
	if maxNeurons <= 0 || neuronsPerCascade <= 0 || data.numData == 0 {
		return // Nothing to do
	}
	
	if data.GetNumInput() != ann.numInput || data.GetNumOutput() != ann.numOutput {
		ann.setError(ErrTrainDataMismatch, "training data size mismatch")
		return
	}

	// Initialize cascade parameters if not set
	if ann.cascadeOutputChangeFraction == 0 {
		ann.SetCascadeOutputChangeFraction(0.01)
	}
	if ann.cascadeOutputStagnationEpochs == 0 {
		ann.SetCascadeOutputStagnationEpochs(12)
	}
	if ann.cascadeCandidateChangeFraction == 0 {
		ann.SetCascadeCandidateChangeFraction(0.01)
	}
	if ann.cascadeCandidateStagnationEpochs == 0 {
		ann.SetCascadeCandidateStagnationEpochs(12)
	}
	if ann.cascadeMaxOutEpochs == 0 {
		ann.SetCascadeMaxOutEpochs(150)
	}
	if ann.cascadeMaxCandEpochs == 0 {
		ann.SetCascadeMaxCandEpochs(150)
	}
	if ann.cascadeMinOutEpochs == 0 {
		ann.SetCascadeMinOutEpochs(50)
	}
	if ann.cascadeMinCandEpochs == 0 {
		ann.SetCascadeMinCandEpochs(50)
	}
	if ann.cascadeCandidateLimit == 0 {
		ann.SetCascadeCandidateLimit(1000.0)
	}
	if len(ann.cascadeActivationFunctions) == 0 {
		// Default activation functions for cascade
		ann.SetCascadeActivationFunctions([]ActivationFunc{
			Sigmoid, SigmoidSymmetric, Gaussian, GaussianSymmetric,
			Elliot, ElliotSymmetric, SinSymmetric, CosSymmetric,
			Sin, Cos,
		})
	}
	if len(ann.cascadeActivationSteepnesses) == 0 {
		// Default steepnesses
		ann.SetCascadeActivationSteepnesses([]T{T(0.25), T(0.5), T(0.75), T(1.0)})
	}

	// Initialize training arrays if needed
	if ann.trainErrors == nil {
		ann.trainErrors = make([]T, ann.totalNeurons)
		ann.trainSlopes = make([]T, ann.totalConnections)
	}

	// Main cascade loop
	neuronsAdded := 0
	for neuronsAdded < maxNeurons && maxNeurons > 0 {
		// Train existing network
		ann.cascadeTrainOutput(data, desiredError)
		
		// Check if we've reached desired error
		mse := ann.TestData(data)
		if mse <= desiredError {
			return
		}
		
		// Check if MSE is getting worse or stuck at 1.0
		if mse >= 0.9999 {
			// Network has saturated, no point adding more neurons
			return
		}

		// Add new neurons
		neuronsToAdd := neuronsPerCascade
		if neuronsAdded + neuronsToAdd > maxNeurons {
			neuronsToAdd = maxNeurons - neuronsAdded
		}

		for i := 0; i < neuronsToAdd; i++ {
			// Train and install best candidate
			ann.cascadeAddCandidate(data, desiredError)
			neuronsAdded++
			
			// Check error after adding neuron
			mse = ann.TestData(data)
			if mse <= desiredError {
				return
			}
			
			// Check callback to see if we should stop
			if ann.callback != nil {
				if !ann.callback(ann, neuronsAdded, float32(mse)) {
					return
				}
			}
		}
		
		// Yield CPU periodically
		runtime.Gosched()
	}
}

// cascadeTrainOutput trains only the output connections
func (ann *Fann[T]) cascadeTrainOutput(data *TrainData[T], desiredError float32) {
	// Initialize training arrays if needed
	if ann.trainErrors == nil {
		ann.trainErrors = make([]T, ann.totalNeurons)
		ann.trainSlopes = make([]T, ann.totalConnections)
	}
	
	// Train output connections
	bestMSE := ann.TestData(data)
	stagnation := 0
	
	for epoch := 0; epoch < ann.cascadeMaxOutEpochs; epoch++ {
		// Must train at least minimum epochs
		if epoch >= ann.cascadeMinOutEpochs {
			// Check for stagnation
			if stagnation >= ann.cascadeOutputStagnationEpochs {
				break
			}
			
			// Check if we've reached desired error
			if bestMSE <= desiredError {
				break
			}
		}
		
		// Train one epoch on output connections only
		ann.cascadeTrainOutputEpoch(data, epoch)
		
		// Calculate MSE
		mse := ann.TestData(data)
		
		// Check for improvement
		if mse < bestMSE * (1.0 - ann.cascadeOutputChangeFraction) {
			bestMSE = mse
			stagnation = 0
		} else {
			stagnation++
		}
		
		// Call callback if set
		if ann.callback != nil {
			if !ann.callback(ann, epoch, float32(mse)) {
				break
			}
		}
		
		// Yield CPU periodically
		if epoch%100 == 0 {
			runtime.Gosched()
		}
	}
}

// cascadeTrainOutputEpoch trains one epoch on output connections
func (ann *Fann[T]) cascadeTrainOutputEpoch(data *TrainData[T], epoch int) {
	if data.numData == 0 {
		return // No data to train on
	}
	
	ann.ResetMSE()
	
	// Clear slopes only for output layer connections
	outputLayer := ann.layers[len(ann.layers)-1]
	for neuronIdx := outputLayer.firstNeuron; neuronIdx < outputLayer.lastNeuron; neuronIdx++ {
		neuron := &ann.neurons[neuronIdx]
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			ann.trainSlopes[i] = 0
		}
	}
	
	for i := 0; i < data.numData; i++ {
		// Forward pass
		output := ann.Run(data.inputs[i])
		if output == nil {
			continue
		}
		
		// Calculate output errors
		for j := 0; j < ann.numOutput; j++ {
			neuronIdx := outputLayer.firstNeuron + j
			neuron := &ann.neurons[neuronIdx]
			
			// Calculate error
			error := data.outputs[i][j] - neuron.value
			
			// Calculate error derivative
			errorDeriv := error * ActivationDerivative(
				neuron.activationFunction,
				neuron.activationSteepness,
				neuron.value,
				neuron.sum,
			)
			
			// Accumulate slopes for this neuron's connections
			for k := neuron.firstCon; k < neuron.lastCon; k++ {
				ann.trainSlopes[k] += errorDeriv * ann.neurons[ann.connections[k]].value
			}
		}
		
		// Update MSE
		ann.calculateMSE(data.outputs[i])
	}
	
	// Update weights based on training algorithm
	switch ann.trainingAlgorithm {
	case TrainIncremental:
		ann.cascadeUpdateWeightsIncremental(outputLayer)
	case TrainBatch:
		ann.cascadeUpdateWeightsBatch(data.numData, outputLayer)
	case TrainRPROP:
		ann.cascadeUpdateWeightsRPROP(data.numData, outputLayer)
	case TrainQuickprop:
		ann.cascadeUpdateWeightsQuickprop(data.numData, outputLayer)
	case TrainSarprop:
		ann.cascadeUpdateWeightsSarprop(data.numData, epoch, outputLayer)
	default:
		// Default to batch for cascade training
		ann.cascadeUpdateWeightsBatch(data.numData, outputLayer)
	}
}

// cascadeUpdateWeightsIncremental updates weights for specific layer using incremental training
func (ann *Fann[T]) cascadeUpdateWeightsIncremental(layer layer[T]) {
	learningRate := ann.learningRate
	
	// Update weights only for specified layer
	for neuronIdx := layer.firstNeuron; neuronIdx < layer.lastNeuron; neuronIdx++ {
		neuron := &ann.neurons[neuronIdx]
		
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			// Incremental update - slopes already contain the gradient for this sample
			ann.weights[i] += T(learningRate) * ann.trainSlopes[i]
		}
	}
}

// cascadeUpdateWeightsBatch updates weights for specific layer using batch training
func (ann *Fann[T]) cascadeUpdateWeightsBatch(numData int, layer layer[T]) {
	learningRate := ann.learningRate
	
	// Update weights only for specified layer
	for neuronIdx := layer.firstNeuron; neuronIdx < layer.lastNeuron; neuronIdx++ {
		neuron := &ann.neurons[neuronIdx]
		
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			// Batch update - average the accumulated slopes
			ann.weights[i] += T(learningRate) * ann.trainSlopes[i] / T(numData)
		}
	}
}

// cascadeUpdateWeightsQuickprop updates weights for specific layer using Quickprop
func (ann *Fann[T]) cascadeUpdateWeightsQuickprop(numData int, layer layer[T]) {
	// For now, fall back to batch - full Quickprop implementation would be complex
	ann.cascadeUpdateWeightsBatch(numData, layer)
}

// cascadeUpdateWeightsSarprop updates weights for specific layer using Sarprop
func (ann *Fann[T]) cascadeUpdateWeightsSarprop(numData int, epoch int, layer layer[T]) {
	// For now, fall back to RPROP - full Sarprop implementation would be complex  
	ann.cascadeUpdateWeightsRPROP(numData, layer)
}

// cascadeUpdateWeightsRPROP updates weights for specific layer using RPROP
func (ann *Fann[T]) cascadeUpdateWeightsRPROP(numData int, layer layer[T]) {
	increaseFactor := ann.rpropIncreaseFactor
	decreaseFactor := ann.rpropDecreaseFactor
	deltaMin := ann.rpropDeltaMin
	deltaMax := ann.rpropDeltaMax
	
	// Initialize RPROP arrays if needed
	if ann.prevSteps == nil {
		ann.prevSteps = make([]T, ann.totalConnections)
		ann.prevTrainSlopes = make([]T, ann.totalConnections)
		for i := range ann.prevSteps {
			ann.prevSteps[i] = T(ann.rpropDeltaZero)
		}
	}
	
	// Update weights only for specified layer
	for neuronIdx := layer.firstNeuron; neuronIdx < layer.lastNeuron; neuronIdx++ {
		neuron := &ann.neurons[neuronIdx]
		
		for i := neuron.firstCon; i < neuron.lastCon; i++ {
			prevSlope := ann.prevTrainSlopes[i]
			slope := ann.trainSlopes[i] / T(numData)
			prevStep := ann.prevSteps[i]
			
			signChange := prevSlope * slope
			
			if signChange >= 0 {
				step := prevStep * T(increaseFactor)
				if step > T(deltaMax) {
					step = T(deltaMax)
				}
				ann.prevSteps[i] = step
				
				if slope > 0 {
					ann.weights[i] -= step
				} else if slope < 0 {
					ann.weights[i] += step
				}
				
				ann.prevTrainSlopes[i] = slope
			} else {
				step := prevStep * T(decreaseFactor)
				if step < T(deltaMin) {
					step = T(deltaMin)
				}
				ann.prevSteps[i] = step
				ann.prevTrainSlopes[i] = 0
			}
		}
	}
}

// cascadeAddCandidate trains and adds the best candidate neuron
func (ann *Fann[T]) cascadeAddCandidate(data *TrainData[T], desiredError float32) {
	// Calculate number of candidate neurons
	numActivationFuncs := len(ann.cascadeActivationFunctions)
	numSteepnesses := len(ann.cascadeActivationSteepnesses)
	numCandidates := numActivationFuncs * numSteepnesses * ann.cascadeNumCandidateGroups
	
	if numCandidates == 0 {
		numCandidates = 1
		ann.cascadeNumCandidateGroups = 1
	}
	
	// Create candidate neurons
	ann.cascadeCandidates = make([]neuron[T], numCandidates)
	ann.cascadeCandidateScores = make([]T, numCandidates)
	
	// Initialize candidates with different activation functions and steepnesses
	candidateIdx := 0
	for g := 0; g < ann.cascadeNumCandidateGroups; g++ {
		for _, activationFunc := range ann.cascadeActivationFunctions {
			for _, steepness := range ann.cascadeActivationSteepnesses {
				if candidateIdx < numCandidates {
					ann.cascadeCandidates[candidateIdx].activationFunction = activationFunc
					ann.cascadeCandidates[candidateIdx].activationSteepness = steepness
					candidateIdx++
				}
			}
		}
	}
	
	// Train candidates
	candidates := ann.cascadeTrainCandidates(data)
	
	// Find best candidate (highest score)
	bestCandidate := 0
	bestScore := candidates[0].score
	for i := 1; i < numCandidates; i++ {
		if candidates[i].score > bestScore {
			bestScore = candidates[i].score
			bestCandidate = i
		}
	}
	
	// Install best candidate with its trained weights
	ann.cascadeInstallCandidate(candidates[bestCandidate])
	
	// After installing the candidate, retrain the output layer to adapt
	ann.cascadeTrainOutput(data, desiredError)
}

// cascadeCandidate represents a candidate neuron with its weights
type cascadeCandidate[T Numeric] struct {
	neuron         neuron[T]
	inputWeights   []T
	outputWeights  []T
	inputSlopes    []T
	outputSlopes   []T
	score          T
}

// cascadeTrainCandidates trains all candidate neurons and returns their weights
func (ann *Fann[T]) cascadeTrainCandidates(data *TrainData[T]) []*cascadeCandidate[T] {
	// Initialize candidate connections
	numCandidates := len(ann.cascadeCandidates)
	numInputConnections := ann.totalNeurons - ann.numOutput
	numOutputConnections := ann.numOutput
	
	// Create candidate structures with weights
	candidates := make([]*cascadeCandidate[T], numCandidates)
	candidateConnections := make([]int, numInputConnections)
	
	// Set up connections (all non-output neurons)
	connIdx := 0
	for i := 0; i < ann.totalNeurons; i++ {
		// Skip output neurons
		isOutput := false
		outputLayer := ann.layers[len(ann.layers)-1]
		if i >= outputLayer.firstNeuron && i < outputLayer.lastNeuron {
			isOutput = true
		}
		
		if !isOutput {
			candidateConnections[connIdx] = i
			connIdx++
		}
	}
	
	// Initialize candidate structures
	for i := 0; i < numCandidates; i++ {
		candidates[i] = &cascadeCandidate[T]{
			neuron:        ann.cascadeCandidates[i],
			inputWeights:  make([]T, numInputConnections),
			outputWeights: make([]T, numOutputConnections),
			inputSlopes:   make([]T, numInputConnections),
			outputSlopes:  make([]T, numOutputConnections),
		}
		
		// Initialize input weights randomly
		for j := 0; j < numInputConnections; j++ {
			randFloat := float64(randomUint64()&0x1FFFFFFFFFFFFF) / float64(1<<53)
			candidates[i].inputWeights[j] = T(-0.1 + randFloat*0.2)
		}
		
		// Initialize output weights using cascade weight multiplier
		for j := 0; j < numOutputConnections; j++ {
			randFloat := float64(randomUint64()&0x1FFFFFFFFFFFFF) / float64(1<<53)
			// Random weight between -multiplier and +multiplier
			candidates[i].outputWeights[j] = ann.cascadeWeightMultiplier * T(2.0*randFloat - 1.0)
		}
	}
	
	// Train candidates
	bestScores := make([]T, numCandidates)
	stagnation := make([]int, numCandidates)
	
	for epoch := 0; epoch < ann.cascadeMaxCandEpochs; epoch++ {
		// Check if all candidates have stagnated
		allStagnated := true
		for i := 0; i < numCandidates; i++ {
			if epoch >= ann.cascadeMinCandEpochs {
				if stagnation[i] < ann.cascadeCandidateStagnationEpochs {
					allStagnated = false
					break
				}
			} else {
				allStagnated = false
				break
			}
		}
		
		if allStagnated {
			break
		}
		
		// Train each candidate
		for candIdx := 0; candIdx < numCandidates; candIdx++ {
			if stagnation[candIdx] >= ann.cascadeCandidateStagnationEpochs {
				continue
			}
			
			score := ann.cascadeTrainCandidateEpoch(data, candidates[candIdx], candidateConnections)
			
			// Check for improvement
			if score > bestScores[candIdx] * T(1.0 + ann.cascadeCandidateChangeFraction) {
				bestScores[candIdx] = score
				stagnation[candIdx] = 0
			} else {
				stagnation[candIdx]++
			}
			
			candidates[candIdx].score = score
		}
		
		// Yield CPU periodically
		if epoch%10 == 0 {
			runtime.Gosched()
		}
	}
	
	return candidates
}

// cascadeTrainCandidateEpoch trains one epoch for a candidate using the FANN algorithm
func (ann *Fann[T]) cascadeTrainCandidateEpoch(data *TrainData[T], 
	candidate *cascadeCandidate[T], connections []int) T {
	
	numInputConnections := len(connections)
	
	// Clear slopes for this epoch
	for i := range candidate.inputSlopes {
		candidate.inputSlopes[i] = 0
	}
	for i := range candidate.outputSlopes {
		candidate.outputSlopes[i] = 0
	}
	
	// Start with sum squared error as initial score (will subtract squared differences)
	// In FANN, MSE_value is actually the sum squared error, not the mean
	totalScore := ann.mse * float32(ann.numMSE)
	
	for i := 0; i < data.numData; i++ {
		// Forward pass through main network
		output := ann.Run(data.inputs[i])
		if output == nil {
			continue
		}
		
		// Calculate candidate activation
		sum := T(0)
		for j := 0; j < numInputConnections; j++ {
			sum += candidate.inputWeights[j] * ann.neurons[connections[j]].value
		}
		
		// Apply activation function
		activation := ann.activation(candidate.neuron.activationFunction, 
			candidate.neuron.activationSteepness, sum)
		
		// Calculate activation derivative
		derivative := ActivationDerivative(candidate.neuron.activationFunction,
			candidate.neuron.activationSteepness, activation, sum)
		
		// Calculate output errors and update slopes
		outputLayer := ann.layers[len(ann.layers)-1]
		errorValue := T(0)
		
		for j := 0; j < ann.numOutput; j++ {
			neuronIdx := outputLayer.firstNeuron + j
			outputError := data.outputs[i][j] - ann.neurons[neuronIdx].value
			
			// Calculate difference: (activation * weight) - error
			// This is what we want to minimize
			diff := (activation * candidate.outputWeights[j]) - outputError
			
			// Update score (subtract squared difference - higher score is better)
			totalScore -= float32(diff * diff)
			
			// Gradient for output weight
			candidate.outputSlopes[j] -= T(2.0) * diff * activation
			
			// Accumulate error for input weight gradients
			errorValue += diff * candidate.outputWeights[j]
		}
		
		// Apply derivative to error
		errorValue *= derivative
		
		// Update input weight gradients
		for j := 0; j < numInputConnections; j++ {
			candidate.inputSlopes[j] -= errorValue * ann.neurons[connections[j]].value
		}
	}
	
	// Update weights based on accumulated slopes
	ann.updateCandidateWeights(candidate, data.numData)
	
	return T(totalScore / float32(data.numData))
}

// updateCandidateWeights updates the weights for a candidate neuron
func (ann *Fann[T]) updateCandidateWeights(candidate *cascadeCandidate[T], numData int) {
	// For cascade training, we typically use RPROP, Quickprop, or Sarprop
	// For simplicity, let's use a basic gradient descent for now
	// TODO: Implement proper RPROP/Quickprop for candidates
	
	learningRate := T(ann.learningRate)
	
	// Update input weights
	for i := range candidate.inputWeights {
		gradient := candidate.inputSlopes[i] / T(numData)
		candidate.inputWeights[i] += learningRate * gradient
	}
	
	// Update output weights
	for i := range candidate.outputWeights {
		gradient := candidate.outputSlopes[i] / T(numData)
		candidate.outputWeights[i] += learningRate * gradient
	}
}

// cascadeInstallCandidate installs a trained candidate as a new hidden neuron
func (ann *Fann[T]) cascadeInstallCandidate(candidate *cascadeCandidate[T]) {
	// Simplified approach: rebuild the entire network structure
	
	// Check if we need to create a hidden layer
	numLayers := len(ann.layers)
	hasHiddenLayer := numLayers > 2
	
	if !hasHiddenLayer {
		// Create network with hidden layer: input -> hidden -> output
		// Start with copying input layer neurons
		inputLayer := ann.layers[0]
		outputLayer := ann.layers[numLayers-1]
		
		// Create new neuron array
		newTotalNeurons := ann.totalNeurons + 1
		newNeurons := make([]neuron[T], newTotalNeurons)
		
		// Copy input neurons (including bias)
		copy(newNeurons[0:inputLayer.lastNeuron], ann.neurons[0:inputLayer.lastNeuron])
		
		// Add new hidden neuron
		hiddenNeuronIdx := inputLayer.lastNeuron
		newNeurons[hiddenNeuronIdx] = candidate.neuron
		
		// Copy output neurons (shift by 1)
		for i := 0; i < ann.numOutput; i++ {
			newNeurons[hiddenNeuronIdx+1+i] = ann.neurons[outputLayer.firstNeuron+i]
		}
		
		// Create new layer structure
		newLayers := make([]layer[T], 3)
		newLayers[0] = inputLayer // Input layer unchanged
		newLayers[1] = layer[T]{
			firstNeuron: inputLayer.lastNeuron,
			lastNeuron:  inputLayer.lastNeuron + 1,
		}
		newLayers[2] = layer[T]{
			firstNeuron: inputLayer.lastNeuron + 1,
			lastNeuron:  inputLayer.lastNeuron + 1 + ann.numOutput,
		}
		
		// Rebuild connections
		// Count total connections needed
		numInputConnections := inputLayer.lastNeuron // All input neurons to hidden
		numOutputConnections := ann.numOutput * (1 + inputLayer.lastNeuron) // Each output connects to hidden + all inputs
		totalConnections := numInputConnections + numOutputConnections
		
		newConnections := make([]int, totalConnections)
		newWeights := make([]T, totalConnections)
		connIdx := 0
		
		// Hidden neuron connections from all inputs
		newNeurons[hiddenNeuronIdx].firstCon = connIdx
		for i := 0; i < inputLayer.lastNeuron; i++ {
			newConnections[connIdx] = i
			// Use trained input weight from candidate
			if i < len(candidate.inputWeights) {
				newWeights[connIdx] = candidate.inputWeights[i]
			} else {
				// This shouldn't happen if candidate was trained properly
				newWeights[connIdx] = T(0.0)
			}
			connIdx++
		}
		newNeurons[hiddenNeuronIdx].lastCon = connIdx
		
		// Output neuron connections
		for i := 0; i < ann.numOutput; i++ {
			outputNeuronIdx := hiddenNeuronIdx + 1 + i
			newNeurons[outputNeuronIdx].firstCon = connIdx
			
			// Connection from hidden neuron
			newConnections[connIdx] = hiddenNeuronIdx
			// Use trained output weight from candidate
			newWeights[connIdx] = candidate.outputWeights[i]
			connIdx++
			
			// Connections from input neurons (direct shortcut connections)
			for j := 0; j < inputLayer.lastNeuron; j++ {
				newConnections[connIdx] = j
				// Use existing weight if available
				oldNeuronIdx := outputLayer.firstNeuron + i
				oldNeuron := &ann.neurons[oldNeuronIdx]
				weightFound := false
				for k := oldNeuron.firstCon; k < oldNeuron.lastCon; k++ {
					if ann.connections[k] == j {
						newWeights[connIdx] = ann.weights[k]
						weightFound = true
						break
					}
				}
				if !weightFound {
					randFloat := float64(randomUint64()&0x1FFFFFFFFFFFFF) / float64(1<<53)
					newWeights[connIdx] = T(-0.1 + randFloat*0.2)
				}
				connIdx++
			}
			
			newNeurons[outputNeuronIdx].lastCon = connIdx
		}
		
		// Update network
		ann.neurons = newNeurons
		ann.layers = newLayers
		ann.connections = newConnections
		ann.weights = newWeights
		ann.totalNeurons = newTotalNeurons
		ann.totalConnections = connIdx
		
	} else {
		// Add to existing hidden layer
		hiddenLayerIdx := numLayers - 2 // Last hidden layer
		hiddenLayer := ann.layers[hiddenLayerIdx]
		outputLayer := ann.layers[numLayers-1]
		
		// Create new neuron array
		newTotalNeurons := ann.totalNeurons + 1
		newNeurons := make([]neuron[T], newTotalNeurons)
		
		// Copy all neurons up to output layer
		copy(newNeurons[0:outputLayer.firstNeuron], ann.neurons[0:outputLayer.firstNeuron])
		
		// Add new hidden neuron
		newHiddenNeuronIdx := outputLayer.firstNeuron
		newNeurons[newHiddenNeuronIdx] = candidate.neuron
		
		// Copy output neurons (shift by 1)
		for i := 0; i < ann.numOutput; i++ {
			newNeurons[newHiddenNeuronIdx+1+i] = ann.neurons[outputLayer.firstNeuron+i]
		}
		
		// Update layer boundaries
		ann.layers[hiddenLayerIdx].lastNeuron++
		for i := hiddenLayerIdx + 1; i < len(ann.layers); i++ {
			ann.layers[i].firstNeuron++
			ann.layers[i].lastNeuron++
		}
		
		// Calculate connections needed
		numPrevNeurons := newHiddenNeuronIdx // All neurons before this one
		numNewInputConnections := numPrevNeurons // New hidden neuron gets input from all previous
		numNewOutputConnections := ann.numOutput // Each output gets one new connection
		
		// Create new connection arrays
		newTotalConnections := ann.totalConnections + numNewInputConnections + numNewOutputConnections
		newConnections := make([]int, newTotalConnections)
		newWeights := make([]T, newTotalConnections)
		
		// Copy connections for neurons before hidden layer
		connIdx := 0
		for i := 0; i < hiddenLayer.firstNeuron; i++ {
			neuron := &ann.neurons[i]
			for j := neuron.firstCon; j < neuron.lastCon; j++ {
				newConnections[connIdx] = ann.connections[j]
				newWeights[connIdx] = ann.weights[j]
				connIdx++
			}
		}
		
		// Copy connections for existing hidden neurons
		for i := hiddenLayer.firstNeuron; i < hiddenLayer.lastNeuron; i++ {
			neuron := &ann.neurons[i]
			newNeurons[i].firstCon = connIdx
			for j := neuron.firstCon; j < neuron.lastCon; j++ {
				newConnections[connIdx] = ann.connections[j]
				newWeights[connIdx] = ann.weights[j]
				connIdx++
			}
			newNeurons[i].lastCon = connIdx
		}
		
		// Add connections for new hidden neuron
		newNeurons[newHiddenNeuronIdx].firstCon = connIdx
		weightIdx := 0
		for i := 0; i < numPrevNeurons; i++ {
			// Skip output neurons
			isOutput := false
			if i >= outputLayer.firstNeuron && i < outputLayer.lastNeuron {
				isOutput = true
			}
			
			if !isOutput {
				newConnections[connIdx] = i
				// Use trained weight from candidate
				if weightIdx < len(candidate.inputWeights) {
					newWeights[connIdx] = candidate.inputWeights[weightIdx]
					weightIdx++
				} else {
					// This shouldn't happen if candidate was trained properly
					newWeights[connIdx] = T(0.0)
				}
				connIdx++
			}
		}
		newNeurons[newHiddenNeuronIdx].lastCon = connIdx
		
		// Update output neuron connections
		for i := 0; i < ann.numOutput; i++ {
			oldNeuronIdx := outputLayer.firstNeuron + i
			newNeuronIdx := newHiddenNeuronIdx + 1 + i
			oldNeuron := &ann.neurons[oldNeuronIdx]
			
			newNeurons[newNeuronIdx].firstCon = connIdx
			
			// Copy existing connections
			for j := oldNeuron.firstCon; j < oldNeuron.lastCon; j++ {
				newConnections[connIdx] = ann.connections[j]
				newWeights[connIdx] = ann.weights[j]
				connIdx++
			}
			
			// Add connection to new hidden neuron
			newConnections[connIdx] = newHiddenNeuronIdx
			// Use trained output weight from candidate
			newWeights[connIdx] = candidate.outputWeights[i]
			connIdx++
			
			newNeurons[newNeuronIdx].lastCon = connIdx
		}
		
		// Update network
		ann.neurons = newNeurons
		ann.connections = newConnections
		ann.weights = newWeights
		ann.totalNeurons = newTotalNeurons
		ann.totalConnections = connIdx
	}
	
	
	// Reinitialize training arrays
	ann.trainErrors = make([]T, ann.totalNeurons)
	ann.trainSlopes = make([]T, ann.totalConnections)
	ann.prevSteps = make([]T, ann.totalConnections)
	ann.prevTrainSlopes = make([]T, ann.totalConnections)
	
	// Initialize RPROP steps with default values
	for i := range ann.prevSteps {
		ann.prevSteps[i] = T(ann.rpropDeltaZero)
	}
}

// cascadeCreateHiddenLayer creates a new hidden layer
func (ann *Fann[T]) cascadeCreateHiddenLayer() {
	// Insert new layer between input and output
	newLayers := make([]layer[T], len(ann.layers)+1)
	
	// Copy input layer
	newLayers[0] = ann.layers[0]
	
	// Create new hidden layer (initially empty)
	newLayers[1] = layer[T]{
		firstNeuron: ann.layers[0].lastNeuron,
		lastNeuron:  ann.layers[0].lastNeuron,
	}
	
	// Copy remaining layers
	for i := 1; i < len(ann.layers); i++ {
		newLayers[i+1] = ann.layers[i]
	}
	
	ann.layers = newLayers
}

// Cascade training parameter setters/getters

// SetCascadeWeightMultiplier sets the weight multiplier for cascade training
func (ann *Fann[T]) SetCascadeWeightMultiplier(multiplier T) {
	ann.cascadeWeightMultiplier = multiplier
}

// GetCascadeWeightMultiplier returns the weight multiplier for cascade training
func (ann *Fann[T]) GetCascadeWeightMultiplier() T {
	return ann.cascadeWeightMultiplier
}

// SetCascadeOutputChangeFraction sets the cascade output change fraction
func (ann *Fann[T]) SetCascadeOutputChangeFraction(fraction float32) {
	ann.cascadeOutputChangeFraction = fraction
}

// GetCascadeOutputChangeFraction returns the cascade output change fraction
func (ann *Fann[T]) GetCascadeOutputChangeFraction() float32 {
	return ann.cascadeOutputChangeFraction
}

// SetCascadeOutputStagnationEpochs sets the cascade output stagnation epochs
func (ann *Fann[T]) SetCascadeOutputStagnationEpochs(epochs int) {
	ann.cascadeOutputStagnationEpochs = epochs
}

// GetCascadeOutputStagnationEpochs returns the cascade output stagnation epochs
func (ann *Fann[T]) GetCascadeOutputStagnationEpochs() int {
	return ann.cascadeOutputStagnationEpochs
}

// SetCascadeCandidateChangeFraction sets the cascade candidate change fraction
func (ann *Fann[T]) SetCascadeCandidateChangeFraction(fraction float32) {
	ann.cascadeCandidateChangeFraction = fraction
}

// GetCascadeCandidateChangeFraction returns the cascade candidate change fraction
func (ann *Fann[T]) GetCascadeCandidateChangeFraction() float32 {
	return ann.cascadeCandidateChangeFraction
}

// SetCascadeCandidateStagnationEpochs sets the cascade candidate stagnation epochs
func (ann *Fann[T]) SetCascadeCandidateStagnationEpochs(epochs int) {
	ann.cascadeCandidateStagnationEpochs = epochs
}

// GetCascadeCandidateStagnationEpochs returns the cascade candidate stagnation epochs
func (ann *Fann[T]) GetCascadeCandidateStagnationEpochs() int {
	return ann.cascadeCandidateStagnationEpochs
}

// SetCascadeCandidateLimit sets the cascade candidate limit
func (ann *Fann[T]) SetCascadeCandidateLimit(limit float32) {
	ann.cascadeCandidateLimit = int(limit)
}

// GetCascadeCandidateLimit returns the cascade candidate limit
func (ann *Fann[T]) GetCascadeCandidateLimit() float32 {
	return float32(ann.cascadeCandidateLimit)
}

// SetCascadeMaxOutEpochs sets the maximum output training epochs
func (ann *Fann[T]) SetCascadeMaxOutEpochs(epochs int) {
	ann.cascadeMaxOutEpochs = epochs
}

// GetCascadeMaxOutEpochs returns the maximum output training epochs
func (ann *Fann[T]) GetCascadeMaxOutEpochs() int {
	return ann.cascadeMaxOutEpochs
}

// SetCascadeMaxCandEpochs sets the maximum candidate training epochs
func (ann *Fann[T]) SetCascadeMaxCandEpochs(epochs int) {
	ann.cascadeMaxCandEpochs = epochs
}

// GetCascadeMaxCandEpochs returns the maximum candidate training epochs
func (ann *Fann[T]) GetCascadeMaxCandEpochs() int {
	return ann.cascadeMaxCandEpochs
}

// SetCascadeMinOutEpochs sets the minimum output training epochs
func (ann *Fann[T]) SetCascadeMinOutEpochs(epochs int) {
	ann.cascadeMinOutEpochs = epochs
}

// GetCascadeMinOutEpochs returns the minimum output training epochs
func (ann *Fann[T]) GetCascadeMinOutEpochs() int {
	return ann.cascadeMinOutEpochs
}

// SetCascadeMinCandEpochs sets the minimum candidate training epochs
func (ann *Fann[T]) SetCascadeMinCandEpochs(epochs int) {
	ann.cascadeMinCandEpochs = epochs
}

// GetCascadeMinCandEpochs returns the minimum candidate training epochs
func (ann *Fann[T]) GetCascadeMinCandEpochs() int {
	return ann.cascadeMinCandEpochs
}

// SetCascadeActivationFunctions sets the activation functions for cascade
func (ann *Fann[T]) SetCascadeActivationFunctions(functions []ActivationFunc) {
	ann.cascadeActivationFunctions = functions
}

// GetCascadeActivationFunctions returns the activation functions for cascade
func (ann *Fann[T]) GetCascadeActivationFunctions() []ActivationFunc {
	return ann.cascadeActivationFunctions
}

// SetCascadeActivationSteepnesses sets the activation steepnesses for cascade
func (ann *Fann[T]) SetCascadeActivationSteepnesses(steepnesses []T) {
	ann.cascadeActivationSteepnesses = steepnesses
}

// GetCascadeActivationSteepnesses returns the activation steepnesses for cascade
func (ann *Fann[T]) GetCascadeActivationSteepnesses() []T {
	return ann.cascadeActivationSteepnesses
}

// SetCascadeNumCandidateGroups sets the number of candidate groups
func (ann *Fann[T]) SetCascadeNumCandidateGroups(groups int) {
	ann.cascadeNumCandidateGroups = groups
}

// GetCascadeNumCandidateGroups returns the number of candidate groups
func (ann *Fann[T]) GetCascadeNumCandidateGroups() int {
	return ann.cascadeNumCandidateGroups
}

// TrainOnFile trains the network directly from a file without loading all data into memory
func (ann *Fann[T]) TrainOnFile(filename string, maxEpochs int, epochsBetweenReports int, desiredError float32) error {
	data, err := ReadTrainFromFile[T](filename)
	if err != nil {
		return fmt.Errorf("cannot load training data: %w", err)
	}
	ann.TrainOnData(data, maxEpochs, epochsBetweenReports, desiredError)
	return nil
}

// TrainEpochOnFile trains one epoch directly from a file
func (ann *Fann[T]) TrainEpochOnFile(filename string) (float32, error) {
	data, err := ReadTrainFromFile[T](filename)
	if err != nil {
		return 0, fmt.Errorf("cannot load training data: %w", err)
	}
	return ann.TrainEpoch(data), nil
}

// CascadetrainOnFile performs cascade training directly from a file
func (ann *Fann[T]) CascadetrainOnFile(filename string, maxNeurons int, neuronsBetweenReports int, desiredError float32) error {
	data, err := ReadTrainFromFile[T](filename)
	if err != nil {
		return fmt.Errorf("cannot load training data: %w", err)
	}
	ann.CascadetrainOnData(data, maxNeurons, neuronsBetweenReports, desiredError)
	return nil
}