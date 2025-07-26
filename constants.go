package gofann

// Training algorithm constants
const (
	// Default learning rate for gradient descent
	DefaultLearningRate = 0.7
	
	// Default learning momentum
	DefaultLearningMomentum = 0.0
	
	// Connection rate constants
	DefaultConnectionRate = 1.0
	DefaultShortcutConnectionRate = 0.0
	
	// Quickprop constants
	DefaultQuickpropDecay = -0.0001
	DefaultQuickpropMu = 1.75
	
	// RPROP constants
	DefaultRpropIncreaseFactor = 1.2
	DefaultRpropDecreaseFactor = 0.5
	DefaultRpropDeltaMin = 0.000001
	DefaultRpropDeltaMax = 50.0
	DefaultRpropDeltaZero = 0.0125
	
	// SARPROP constants
	DefaultSarpropStepErrorThresholdFactor = 0.1
	DefaultSarpropStepErrorShift = 1.385
	DefaultSarpropTemperature = 0.015
	DefaultSarpropWeightDecayShift = -6.644
	DefaultSarpropEpoch = 1
	
	// Cascade training constants
	DefaultCascadeOutputChangeFraction = 0.01
	DefaultCascadeCandidateChangeFraction = 0.01
	DefaultCascadeOutputStagnationEpochs = 12
	DefaultCascadeCandidateStagnationEpochs = 12
	DefaultCascadeWeightMultiplier = 0.4
	DefaultCascadeCandidateLimit = 1000.0
	DefaultCascadeMaxOutEpochs = 150
	DefaultCascadeMaxCandEpochs = 150
	DefaultCascadeMinOutEpochs = 50
	DefaultCascadeMinCandEpochs = 50
	DefaultCascadeNumCandidateGroups = 2
	
	// Activation steepness
	DefaultActivationSteepnessHidden = 0.5
	DefaultActivationSteepnessOutput = 0.5
	
	// Training constants
	DefaultBitFailLimit = 0.35
	
	// Small epsilon values for numerical stability
	EpsilonQuickprop = 0.000001
	EpsilonRprop = 0.0001
	EpsilonDefault = 0.000001
	
	// Maximum safe integer for random seed
	MaxSafeInteger = 1 << 53
	
	// Reflective training constants
	DefaultTargetAccuracy = 0.95
	DefaultImprovementThreshold = 0.01
	DefaultMaxReflectionCycles = 100
	DefaultLearningRateDecayFactor = 0.95
	DefaultPlateauPatience = 3
	DefaultWeaknessThreshold = 0.1
	DefaultMaxWeaknesses = 5
	DefaultSamplesPerWeakness = 10
	DefaultDiversityFactor = 0.2
	
	// MoE router constants
	DefaultDiversityBonus = 0.1
	DefaultConfidenceWeightFactor = 0.7
	DefaultDomainWeightFactor = 0.3
	DefaultAdaptationRate = 0.1
)

// Fixed-point arithmetic constants
const (
	DefaultFixedPointDecimals = 10
	FixedPointMultiplier = 1024 // 2^10
)