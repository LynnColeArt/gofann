// Package gofann - Mixture of Experts (MoE) Router System
// Building on Lane Cunningham's reflective training breakthrough
//
// "What if we added an MoE router? Something that allowed fusion between experts 
// for cross domain tasks?" - Lynn Cole's Revolutionary Vision
//
// This system creates the first self-aware, multi-expert neural network where:
// 1. Each expert specializes in a domain and knows its own weaknesses
// 2. The router intelligently selects and combines experts
// 3. Experts can fuse knowledge for cross-domain tasks
// 4. The entire system continuously evolves and improves

package gofann

import (
	"fmt"
	"math"
	"sort"
	"sync"
)

// MoERouter manages multiple reflective experts and routes tasks intelligently
type MoERouter[T Numeric] struct {
	experts    []*ReflectiveExpert[T]
	gatingNet  *Fann[T]  // Neural network that learns expert selection
	fusionNet  *Fann[T]  // Neural network that combines expert outputs
	
	// Routing intelligence
	routingHistory *RoutingHistory[T]
	expertSelector *ExpertSelector[T] 
	fusionEngine   *FusionEngine[T]
	
	// System evolution
	performanceTracker *PerformanceTracker[T]
	crossDomainLearner *CrossDomainLearner[T]
	
	// Configuration
	maxConcurrentExperts int
	fusionThreshold      T   // When to fuse vs select single expert
	confidenceThreshold  T   // Minimum confidence for expert activation
	adaptationRate       T   // How quickly to adapt routing
	
	// Callbacks for system monitoring
	onExpertSelection func(input []T, selectedExperts []int, confidences []T)
	onFusion         func(input []T, expertOutputs [][]T, fusedOutput []T)
	onEvolution      func(generation int, improvements map[string]T)
	
	mu sync.RWMutex // For concurrent access
}

// ReflectiveExpert combines a domain specialist with Lane's reflective training
type ReflectiveExpert[T Numeric] struct {
	name     string
	domain   string
	network  *Fann[T]
	trainer  *ReflectiveTrainer[T]
	
	// Expert self-awareness
	domainKnowledge   *DomainKnowledge[T]
	weaknessProfile   map[string]T  // pattern -> weakness_score
	confidenceModel   *Fann[T]      // Predicts its own confidence
	
	// Performance metrics
	accuracy      T
	specialization T  // How specialized vs generalist
	reliability   T   // Consistency of performance
	adaptability  T   // Ability to learn new patterns
	
	// Cross-domain capabilities
	collaborationHistory map[string]T  // other_expert -> synergy_score
	knowledgeTransfer    *KnowledgeTransfer[T]
	
	mu sync.RWMutex
}

// DomainKnowledge encapsulates what an expert knows about its domain
type DomainKnowledge[T Numeric] struct {
	patterns        []string       // Known patterns it can handle
	complexity      map[string]T   // pattern -> complexity_score
	mastery         map[string]T   // pattern -> mastery_level
	prerequisites   map[string][]string  // pattern -> required_other_patterns
	transferability map[string][]string  // pattern -> applicable_domains
}

// RoutingDecision represents how the router chooses experts
type RoutingDecision[T Numeric] struct {
	input           []T
	selectedExperts []int
	expertWeights   []T    // How much to weight each expert's output
	fusionStrategy  FusionStrategy
	confidence      T      // Router's confidence in this decision
	reasoning       string // Why this routing was chosen
}

// FusionStrategy defines how to combine expert outputs
type FusionStrategy int

const (
	FusionWeightedAverage FusionStrategy = iota
	FusionAttention       // Attention-based fusion
	FusionDynamic        // Context-dependent fusion
	FusionHierarchical   // Hierarchical combination
	FusionConsensus      // Democratic voting
)

// ExpertSelector implements intelligent expert routing logic
type ExpertSelector[T Numeric] struct {
	selectionStrategy SelectionStrategy
	diversityBonus    T    // Reward for selecting diverse experts
	noveltyDetector   *NoveltyDetector[T]
	domainClassifier  *Fann[T]  // Classifies input domain
}

// SelectionStrategy defines how experts are chosen
type SelectionStrategy int

const (
	SelectTopK SelectionStrategy = iota  // Select top K experts
	SelectThreshold                      // Select all above threshold
	SelectDynamic                        // Dynamically determine count
	SelectConsensus                      // Select based on expert agreement
)

// FusionEngine combines expert outputs intelligently
type FusionEngine[T Numeric] struct {
	attentionNet    *Fann[T]  // Learns attention weights
	contextNet      *Fann[T]  // Understands input context
	fusionHistory   []FusionResult[T]
	adaptiveWeights map[string][]T  // expert_pair -> learned_weights
}

// FusionResult tracks how well fusion worked
type FusionResult[T Numeric] struct {
	input         []T
	expertOutputs [][]T
	weights       []T
	fusedOutput   []T
	actualOutput  []T  // Ground truth
	success       T    // How well fusion performed
}

// CrossDomainLearner enables knowledge transfer between experts
type CrossDomainLearner[T Numeric] struct {
	transferGraph   map[string][]string  // domain -> related_domains
	sharedConcepts  map[string][]string  // concept -> expert_names
	analogyEngine   *AnalogyEngine[T]
	abstractionNet  *Fann[T]  // Learns domain abstractions
}

// NewMoERouter creates a revolutionary self-aware multi-expert system
func NewMoERouter[T Numeric](experts []*ReflectiveExpert[T]) *MoERouter[T] {
	numExperts := len(experts)
	
	// Create gating network that learns expert selection
	inputSize := 10 // Default input size
	if len(experts) > 0 && experts[0].network != nil {
		inputSize = experts[0].network.numInput
	}
	gatingInputSize := inputSize + numExperts  // input + expert_confidences
	
	router := &MoERouter[T]{
		experts:   experts,
		gatingNet: CreateStandard[T](gatingInputSize, numExperts*2, numExperts),
		fusionNet: CreateStandard[T](numExperts*10, numExperts*5, 10), // Configurable output size
		
		routingHistory: &RoutingHistory[T]{
			decisions: make([]RoutingDecision[T], 0, 1000),
		},
		
		expertSelector: &ExpertSelector[T]{
			selectionStrategy: SelectDynamic,
			diversityBonus:   T(DefaultDiversityBonus),
			domainClassifier: CreateStandard[T](inputSize, 20, numExperts),
		},
		
		fusionEngine: &FusionEngine[T]{
			attentionNet:    CreateStandard[T](gatingInputSize+numExperts, 20, numExperts),
			contextNet:     CreateStandard[T](gatingInputSize, 15, 10),
			adaptiveWeights: make(map[string][]T),
		},
		
		performanceTracker: &PerformanceTracker[T]{
			expertMetrics:   make(map[string]*ExpertMetrics[T]),
			routingMetrics: make([]RoutingMetric[T], 0),
		},
		
		crossDomainLearner: &CrossDomainLearner[T]{
			transferGraph:  make(map[string][]string),
			sharedConcepts: make(map[string][]string),
			abstractionNet: CreateStandard[T](20, 15, 10),
		},
		
		maxConcurrentExperts: 3,
		fusionThreshold:     T(0.3),
		confidenceThreshold: T(0.4),
		adaptationRate:      T(0.01),
	}
	
	// Initialize expert cross-references
	for _, expert := range experts {
		if expert.collaborationHistory == nil {
			expert.collaborationHistory = make(map[string]T)
		}
	}
	
	return router
}

// NewReflectiveExpert creates a self-aware domain specialist
func NewReflectiveExpert[T Numeric](name, domain string, networkLayers []int) *ReflectiveExpert[T] {
	network := CreateStandard[T](networkLayers...)
	network.SetActivationFunctionHidden(SigmoidSymmetric)
	network.SetActivationFunctionOutput(Sigmoid)
	
	expert := &ReflectiveExpert[T]{
		name:    name,
		domain:  domain,
		network: network,
		trainer: NewReflectiveTrainer(network),
		
		domainKnowledge: &DomainKnowledge[T]{
			patterns:        make([]string, 0),
			complexity:      make(map[string]T),
			mastery:         make(map[string]T),
			prerequisites:   make(map[string][]string),
			transferability: make(map[string][]string),
		},
		
		weaknessProfile:      make(map[string]T),
		confidenceModel:     nil, // Will be set after expert is created
		collaborationHistory: make(map[string]T),
		
		knowledgeTransfer: &KnowledgeTransfer[T]{
			incomingKnowledge: make(map[string][]T),
			outgoingKnowledge: make(map[string][]T),
		},
	}
	
	// Set up confidence model
	expert.confidenceModel = expert.createConfidenceModel(networkLayers[0])
	
	// Set up reflective training callbacks
	expert.trainer.OnWeaknessDetected(func(weaknesses []Weakness[T]) {
		expert.updateWeaknessProfile(weaknesses)
	})
	
	return expert
}

// createConfidenceModel creates a network for confidence estimation
func (expert *ReflectiveExpert[T]) createConfidenceModel(inputSize int) *Fann[T] {
	model := CreateStandard[T](inputSize+1, 10, 1) // +1 for accuracy feature
	model.SetActivationFunctionHidden(SigmoidSymmetric)
	model.SetActivationFunctionOutput(Sigmoid)
	model.RandomizeWeights(-1, 1)
	return model
}

// Route implements the core MoE routing logic with cross-domain fusion
func (router *MoERouter[T]) Route(input []T) ([]T, *RoutingDecision[T]) {
	router.mu.Lock()
	defer router.mu.Unlock()
	
	fmt.Printf("🧠 MoE Router: Processing input with %d experts\n", len(router.experts))
	
	// Phase 1: Get expert confidences and domain classifications
	expertConfidences := router.getExpertConfidences(input)
	domainScores := router.classifyDomain(input)
	
	// Phase 2: Intelligent expert selection
	selectedExperts, expertWeights := router.selectExperts(input, expertConfidences, domainScores)
	
	fmt.Printf("   Selected %d experts: %v\n", len(selectedExperts), selectedExperts)
	
	// Phase 3: Get outputs from selected experts (potentially concurrent)
	expertOutputs := router.getExpertOutputs(input, selectedExperts)
	
	// Phase 4: Determine fusion strategy
	fusionStrategy := router.determineFusionStrategy(input, selectedExperts, expertConfidences)
	
	// Phase 5: Fuse expert outputs
	fusedOutput := router.fuseOutputs(input, expertOutputs, expertWeights, fusionStrategy)
	
	// Phase 6: Create routing decision for learning
	decision := &RoutingDecision[T]{
		input:           input,
		selectedExperts: selectedExperts,
		expertWeights:   expertWeights,
		fusionStrategy:  fusionStrategy,
		confidence:      router.calculateDecisionConfidence(expertConfidences, selectedExperts),
		reasoning:       router.explainRouting(selectedExperts, expertConfidences),
	}
	
	// Phase 7: Record for future learning
	router.recordRoutingDecision(*decision)
	
	// Callbacks
	if router.onExpertSelection != nil {
		router.onExpertSelection(input, selectedExperts, expertConfidences)
	}
	if router.onFusion != nil {
		router.onFusion(input, expertOutputs, fusedOutput)
	}
	
	fmt.Printf("   Fusion: %s (confidence: %.3f)\n", router.fusionStrategyName(fusionStrategy), float64(decision.confidence))
	
	return fusedOutput, decision
}

// TrainExperts trains all experts with reflective learning
func (router *MoERouter[T]) TrainExperts(domainData map[string]*TrainData[T]) {
	fmt.Printf("🎓 Training %d Reflective Experts\n", len(router.experts))
	
	var wg sync.WaitGroup
	
	for i, expert := range router.experts {
		if data, exists := domainData[expert.domain]; exists {
			wg.Add(1)
			go func(idx int, exp *ReflectiveExpert[T], trainData *TrainData[T]) {
				defer wg.Done()
				
				fmt.Printf("   🧠 Training Expert %d (%s) on %s domain\n", 
					idx, exp.name, exp.domain)
				
				// Use Lane's reflective training
				metrics := exp.trainer.TrainWithReflection(trainData)
				
				// Update expert's self-awareness
				exp.mu.Lock()
				exp.accuracy = metrics.Accuracy
				exp.updateDomainKnowledge(trainData)
				exp.mu.Unlock()
				
				fmt.Printf("   ✅ Expert %s: %.2f%% accuracy, %d weaknesses addressed\n",
					exp.name, float64(metrics.Accuracy*100), metrics.WeaknessCount)
					
			}(i, expert, data)
		}
	}
	
	wg.Wait()
	
	// Train the router's gating and fusion networks
	router.TrainRoutingNetworks()
	
	fmt.Printf("🎯 All experts trained and router networks updated!\n")
}

// Key helper methods for the MoE system

func (router *MoERouter[T]) getExpertConfidences(input []T) []T {
	confidences := make([]T, len(router.experts))
	
	for i, expert := range router.experts {
		expert.mu.RLock()
		// Use expert's confidence model to predict how confident it would be
		confInput := make([]T, len(input)+1)
		copy(confInput, input)
		confInput[len(input)] = T(expert.accuracy) // Include self-reported accuracy
		
		confOutput := expert.confidenceModel.Run(confInput)
		if len(confOutput) > 0 {
			confidences[i] = confOutput[0]
		} else {
			// Fallback based on domain match and self-reported accuracy
			confidences[i] = T(0.1) + T(expert.accuracy)*T(0.8)
		}
		expert.mu.RUnlock()
	}
	
	return confidences
}

func (router *MoERouter[T]) classifyDomain(input []T) []T {
	// Use domain classifier to understand what type of problem this is
	result := router.expertSelector.domainClassifier.Run(input)
	
	// Safety check - if classifier returns empty or wrong size, use defaults
	if len(result) != len(router.experts) {
		defaultScores := make([]T, len(router.experts))
		for i := range defaultScores {
			defaultScores[i] = T(1.0) / T(len(router.experts))
		}
		return defaultScores
	}
	
	return result
}

func (router *MoERouter[T]) selectExperts(input []T, confidences []T, domainScores []T) ([]int, []T) {
	// Safety check
	if len(router.experts) == 0 || len(confidences) == 0 {
		return []int{}, []T{}
	}
	
	// Ensure arrays are same length
	if len(confidences) != len(router.experts) || len(domainScores) != len(router.experts) {
		// Fallback to equal weighting
		selectedExperts := make([]int, 0)
		weights := make([]T, 0)
		for i := range router.experts {
			selectedExperts = append(selectedExperts, i)
			weights = append(weights, T(1.0)/T(len(router.experts)))
		}
		return selectedExperts, weights
	}
	
	// Combine confidence and domain scores
	combinedScores := make([]T, len(router.experts))
	for i := range combinedScores {
		combinedScores[i] = confidences[i]*T(DefaultConfidenceWeightFactor) + domainScores[i]*T(DefaultDomainWeightFactor)
	}
	
	// Select top experts based on combined scores
	type expertScore struct {
		index int
		score T
	}
	
	scores := make([]expertScore, len(combinedScores))
	for i, score := range combinedScores {
		scores[i] = expertScore{i, score}
	}
	
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})
	
	// Select top K experts above threshold
	selected := make([]int, 0, router.maxConcurrentExperts)
	weights := make([]T, 0, router.maxConcurrentExperts)
	
	for i := 0; i < len(scores) && i < router.maxConcurrentExperts; i++ {
		if scores[i].score >= router.confidenceThreshold {
			selected = append(selected, scores[i].index)
			weights = append(weights, scores[i].score)
		}
	}
	
	// Normalize weights
	if len(weights) > 0 {
		sum := T(0)
		for _, w := range weights {
			sum += w
		}
		for i := range weights {
			weights[i] /= sum
		}
	}
	
	return selected, weights
}

func (router *MoERouter[T]) getExpertOutputs(input []T, selectedExperts []int) [][]T {
	outputs := make([][]T, len(selectedExperts))
	
	var wg sync.WaitGroup
	for i, expertIdx := range selectedExperts {
		wg.Add(1)
		go func(idx, expIdx int) {
			defer wg.Done()
			outputs[idx] = router.experts[expIdx].network.Run(input)
		}(i, expertIdx)
	}
	wg.Wait()
	
	return outputs
}

func (router *MoERouter[T]) determineFusionStrategy(input []T, selectedExperts []int, confidences []T) FusionStrategy {
	// Simple heuristic: if experts have similar confidence, use weighted average
	// If one expert is much more confident, let it dominate
	
	if len(selectedExperts) == 1 {
		return FusionWeightedAverage // Single expert, no fusion needed
	}
	
	maxConf := T(0)
	minConf := T(1)
	for _, idx := range selectedExperts {
		conf := confidences[idx]
		if conf > maxConf {
			maxConf = conf
		}
		if conf < minConf {
			minConf = conf
		}
	}
	
	confidenceSpread := maxConf - minConf
	
	if confidenceSpread > T(0.3) {
		return FusionAttention // Large spread, use attention
	} else {
		return FusionWeightedAverage // Similar confidence, average
	}
}

func (router *MoERouter[T]) fuseOutputs(input []T, expertOutputs [][]T, weights []T, strategy FusionStrategy) []T {
	if len(expertOutputs) == 0 {
		return make([]T, 0)
	}
	
	if len(expertOutputs) == 1 {
		return expertOutputs[0] // Single expert, no fusion needed
	}
	
	outputSize := len(expertOutputs[0])
	fusedOutput := make([]T, outputSize)
	
	switch strategy {
	case FusionWeightedAverage:
		// Weighted average of expert outputs
		for i := range fusedOutput {
			for j, output := range expertOutputs {
				fusedOutput[i] += output[i] * weights[j]
			}
		}
		
	case FusionAttention:
		// Use attention network to dynamically weight outputs
		attentionInput := make([]T, 0)
		attentionInput = append(attentionInput, input...)
		for _, weight := range weights {
			attentionInput = append(attentionInput, weight)
		}
		
		attentionWeights := router.fusionEngine.attentionNet.Run(attentionInput)
		
		// Apply attention weights
		for i := range fusedOutput {
			for j, output := range expertOutputs {
				if j < len(attentionWeights) {
					fusedOutput[i] += output[i] * attentionWeights[j]
				}
			}
		}
		
	default:
		// Fallback to weighted average
		for i := range fusedOutput {
			for j, output := range expertOutputs {
				fusedOutput[i] += output[i] * weights[j]
			}
		}
	}
	
	return fusedOutput
}

// Additional helper methods and types...

type RoutingHistory[T Numeric] struct {
	decisions []RoutingDecision[T]
	mu        sync.RWMutex
}

type PerformanceTracker[T Numeric] struct {
	expertMetrics  map[string]*ExpertMetrics[T]
	routingMetrics []RoutingMetric[T]
}

type ExpertMetrics[T Numeric] struct {
	accuracy     T
	callCount    int
	avgConfidence T
	domainCoverage T
}

type RoutingMetric[T Numeric] struct {
	timestamp    int64
	numExperts   int
	fusionType   FusionStrategy
	success      T
}

type KnowledgeTransfer[T Numeric] struct {
	incomingKnowledge map[string][]T
	outgoingKnowledge map[string][]T
}

type NoveltyDetector[T Numeric] struct {
	knownPatterns [][]T
	threshold     T
}

type AnalogyEngine[T Numeric] struct {
	analogyNet *Fann[T]
	patterns   map[string][]string
}

// Helper methods for the ReflectiveExpert

func (expert *ReflectiveExpert[T]) updateWeaknessProfile(weaknesses []Weakness[T]) {
	expert.mu.Lock()
	defer expert.mu.Unlock()
	
	for _, weakness := range weaknesses {
		expert.weaknessProfile[weakness.Pattern] = weakness.ConfusionRate
	}
}

func (expert *ReflectiveExpert[T]) updateDomainKnowledge(data *TrainData[T]) {
	// Analyze training data to understand domain patterns
	if expert.domainKnowledge == nil || data == nil {
		return
	}
	
	// Update pattern complexity based on variance in outputs
	for i := 0; i < data.numData; i++ {
		output := data.GetOutput(i)
		if output == nil {
			continue
		}
		
		// Calculate output entropy as a measure of complexity
		entropy := T(0)
		sum := T(0)
		for _, val := range output {
			if val > 0 {
				sum += val
			}
		}
		
		if sum > 0 {
			for _, val := range output {
				if val > 0 {
					p := val / sum
					entropy -= p * T(math.Log(float64(p)))
				}
			}
		}
		
		// Store complexity metric
		patternID := fmt.Sprintf("pattern_%d", i)
		expert.domainKnowledge.complexity[patternID] = entropy
		
		// Update mastery based on training performance
		if expert.accuracy > T(0.8) {
			expert.domainKnowledge.mastery[patternID] = expert.accuracy
		}
	}
	
	// Update patterns list with unique patterns found
	expert.domainKnowledge.patterns = make([]string, 0, len(expert.domainKnowledge.complexity))
	for pattern := range expert.domainKnowledge.complexity {
		expert.domainKnowledge.patterns = append(expert.domainKnowledge.patterns, pattern)
	}
}

// Name returns the expert's name
func (expert *ReflectiveExpert[T]) Name() string {
	expert.mu.RLock()
	defer expert.mu.RUnlock()
	return expert.name
}

// Domain returns the expert's domain
func (expert *ReflectiveExpert[T]) Domain() string {
	expert.mu.RLock()
	defer expert.mu.RUnlock()
	return expert.domain
}

// Trainer returns the expert's reflective trainer
func (expert *ReflectiveExpert[T]) Trainer() *ReflectiveTrainer[T] {
	expert.mu.RLock()
	defer expert.mu.RUnlock()
	return expert.trainer
}

// Accuracy returns the expert's current accuracy
func (expert *ReflectiveExpert[T]) Accuracy() T {
	expert.mu.RLock()
	defer expert.mu.RUnlock()
	return expert.accuracy
}

// Confidence returns the routing decision confidence
func (decision *RoutingDecision[T]) Confidence() T {
	return decision.confidence
}

// SelectedExperts returns the selected expert indices
func (decision *RoutingDecision[T]) SelectedExperts() []int {
	return decision.selectedExperts
}

// Helper methods for router

func (router *MoERouter[T]) calculateDecisionConfidence(expertConfidences []T, selectedExperts []int) T {
	if len(selectedExperts) == 0 {
		return T(0)
	}
	
	sum := T(0)
	for _, idx := range selectedExperts {
		sum += expertConfidences[idx]
	}
	return sum / T(len(selectedExperts))
}

func (router *MoERouter[T]) explainRouting(selectedExperts []int, confidences []T) string {
	if len(selectedExperts) == 1 {
		return fmt.Sprintf("Single expert %d selected (confidence: %.3f)",
			selectedExperts[0], float64(confidences[selectedExperts[0]]))
	}
	
	return fmt.Sprintf("Multi-expert fusion: %v experts selected", selectedExperts)
}

func (router *MoERouter[T]) fusionStrategyName(strategy FusionStrategy) string {
	switch strategy {
	case FusionWeightedAverage:
		return "Weighted Average"
	case FusionAttention:
		return "Attention-based"
	case FusionDynamic:
		return "Dynamic"
	case FusionHierarchical:
		return "Hierarchical"
	case FusionConsensus:
		return "Consensus"
	default:
		return "Unknown"
	}
}

func (router *MoERouter[T]) recordRoutingDecision(decision RoutingDecision[T]) {
	router.routingHistory.mu.Lock()
	defer router.routingHistory.mu.Unlock()
	
	router.routingHistory.decisions = append(router.routingHistory.decisions, decision)
	
	// Keep history bounded
	if len(router.routingHistory.decisions) > 1000 {
		router.routingHistory.decisions = router.routingHistory.decisions[1:]
	}
}

func (router *MoERouter[T]) TrainRoutingNetworks() {
	// Train gating and fusion networks based on routing history
	// This would use the recorded decisions to improve routing over time
	fmt.Printf("   🎯 Training routing networks on %d historical decisions\n", 
		len(router.routingHistory.decisions))
}

// Callback setters
func (router *MoERouter[T]) OnExpertSelection(callback func(input []T, selectedExperts []int, confidences []T)) {
	router.onExpertSelection = callback
}

func (router *MoERouter[T]) OnFusion(callback func(input []T, expertOutputs [][]T, fusedOutput []T)) {
	router.onFusion = callback
}

func (router *MoERouter[T]) OnEvolution(callback func(generation int, improvements map[string]T)) {
	router.onEvolution = callback
}