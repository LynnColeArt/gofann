package gofann

// Copy creates a deep copy of the neural network
func (ann *Fann[T]) Copy() *Fann[T] {
	if ann == nil {
		return nil
	}
	
	// Create new network structure
	netCopy := &Fann[T]{
		// Error handling
		errno:  ann.errno,
		errstr: ann.errstr,
		
		// Network topology
		networkType:      ann.networkType,
		connectionRate:   ann.connectionRate,
		numInput:         ann.numInput,
		numOutput:        ann.numOutput,
		totalNeurons:     ann.totalNeurons,
		totalConnections: ann.totalConnections,
		
		// Training parameters
		trainingAlgorithm:  ann.trainingAlgorithm,
		learningRate:       ann.learningRate,
		learningMomentum:   ann.learningMomentum,
		trainErrorFunction: ann.trainErrorFunction,
		trainStopFunction:  ann.trainStopFunction,
		bitFailLimit:       ann.bitFailLimit,
		
		// RPROP parameters
		rpropIncreaseFactor: ann.rpropIncreaseFactor,
		rpropDecreaseFactor: ann.rpropDecreaseFactor,
		rpropDeltaMin:      ann.rpropDeltaMin,
		rpropDeltaMax:      ann.rpropDeltaMax,
		rpropDeltaZero:     ann.rpropDeltaZero,
		
		// Quickprop parameters
		quickpropDecay: ann.quickpropDecay,
		quickpropMu:    ann.quickpropMu,
		
		// Sarprop parameters
		sarpropWeightDecayShift:         ann.sarpropWeightDecayShift,
		sarpropStepErrorThresholdFactor: ann.sarpropStepErrorThresholdFactor,
		sarpropStepErrorShift:           ann.sarpropStepErrorShift,
		sarpropTemperature:              ann.sarpropTemperature,
		sarpropEpoch:                    ann.sarpropEpoch,
		
		// Cascade parameters
		cascadeOutputChangeFraction:      ann.cascadeOutputChangeFraction,
		cascadeOutputStagnationEpochs:    ann.cascadeOutputStagnationEpochs,
		cascadeCandidateChangeFraction:   ann.cascadeCandidateChangeFraction,
		cascadeCandidateStagnationEpochs: ann.cascadeCandidateStagnationEpochs,
		cascadeCandidateLimit:            ann.cascadeCandidateLimit,
		cascadeMaxOutEpochs:              ann.cascadeMaxOutEpochs,
		cascadeMaxCandEpochs:             ann.cascadeMaxCandEpochs,
		cascadeMinOutEpochs:              ann.cascadeMinOutEpochs,
		cascadeMinCandEpochs:             ann.cascadeMinCandEpochs,
		cascadeNumCandidateGroups:        ann.cascadeNumCandidateGroups,
		cascadeWeightMultiplier:          ann.cascadeWeightMultiplier,
		
		// MSE calculation
		mse:     ann.mse,
		numMSE:  ann.numMSE,
		bitFail: ann.bitFail,
		
		// Note: callback is not copied as it may contain closure-specific state
	}
	
	// Deep copy layers
	netCopy.layers = make([]layer[T], len(ann.layers))
	copy(netCopy.layers, ann.layers)
	
	// Deep copy neurons
	netCopy.neurons = make([]neuron[T], len(ann.neurons))
	copy(netCopy.neurons, ann.neurons)
	
	// Deep copy weights
	netCopy.weights = make([]T, len(ann.weights))
	copy(netCopy.weights, ann.weights)
	
	// Deep copy connections
	netCopy.connections = make([]int, len(ann.connections))
	copy(netCopy.connections, ann.connections)
	
	// Deep copy training state arrays if they exist
	if ann.trainErrors != nil {
		netCopy.trainErrors = make([]T, len(ann.trainErrors))
		copy(netCopy.trainErrors, ann.trainErrors)
	}
	if ann.trainSlopes != nil {
		netCopy.trainSlopes = make([]T, len(ann.trainSlopes))
		copy(netCopy.trainSlopes, ann.trainSlopes)
	}
	if ann.prevSteps != nil {
		netCopy.prevSteps = make([]T, len(ann.prevSteps))
		copy(netCopy.prevSteps, ann.prevSteps)
	}
	if ann.prevTrainSlopes != nil {
		netCopy.prevTrainSlopes = make([]T, len(ann.prevTrainSlopes))
		copy(netCopy.prevTrainSlopes, ann.prevTrainSlopes)
	}
	if ann.prevWeightDeltas != nil {
		netCopy.prevWeightDeltas = make([]T, len(ann.prevWeightDeltas))
		copy(netCopy.prevWeightDeltas, ann.prevWeightDeltas)
	}
	
	// Deep copy cascade candidates if they exist
	if ann.cascadeCandidates != nil {
		netCopy.cascadeCandidates = make([]neuron[T], len(ann.cascadeCandidates))
		copy(netCopy.cascadeCandidates, ann.cascadeCandidates)
	}
	if ann.cascadeCandidateScores != nil {
		netCopy.cascadeCandidateScores = make([]T, len(ann.cascadeCandidateScores))
		copy(netCopy.cascadeCandidateScores, ann.cascadeCandidateScores)
	}
	
	// Deep copy cascade activation functions
	if ann.cascadeActivationFunctions != nil {
		netCopy.cascadeActivationFunctions = make([]ActivationFunc, len(ann.cascadeActivationFunctions))
		copy(netCopy.cascadeActivationFunctions, ann.cascadeActivationFunctions)
	}
	
	// Deep copy cascade activation steepnesses
	if ann.cascadeActivationSteepnesses != nil {
		netCopy.cascadeActivationSteepnesses = make([]T, len(ann.cascadeActivationSteepnesses))
		copy(netCopy.cascadeActivationSteepnesses, ann.cascadeActivationSteepnesses)
	}
	
	// Deep copy scale parameters
	if ann.scaleParams != nil {
		netCopy.scaleParams = &ScaleParams[T]{
			enabled: ann.scaleParams.enabled,
		}
		
		// Copy input scale
		if ann.scaleParams.inputScale != nil {
			netCopy.scaleParams.inputScale = make([]ScaleInput[T], len(ann.scaleParams.inputScale))
			copy(netCopy.scaleParams.inputScale, ann.scaleParams.inputScale)
		}
		
		// Copy output scale
		if ann.scaleParams.outputScale != nil {
			netCopy.scaleParams.outputScale = make([]ScaleOutput[T], len(ann.scaleParams.outputScale))
			copy(netCopy.scaleParams.outputScale, ann.scaleParams.outputScale)
		}
	}
	
	return netCopy
}