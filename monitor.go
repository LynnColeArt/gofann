// Package gofann - Training Monitor and Visualization
// Provides real-time monitoring and visualization of neural network training
package gofann

import (
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"
	"time"
)

// TrainingMonitor tracks and visualizes training progress
type TrainingMonitor[T Numeric] struct {
	mu             sync.RWMutex
	startTime      time.Time
	currentEpoch   int
	totalEpochs    int
	currentMSE     float32
	bestMSE        float32
	mseHistory     []float32
	accuracyHistory []float32
	learningRates  []float32
	
	// Visualization settings
	width          int
	height         int
	output         io.Writer
	updateInterval time.Duration
	lastUpdate     time.Time
	
	// Callbacks
	onEpoch        func(epoch int, mse float32)
	onImprovement  func(oldBest, newBest float32)
	onPlateau      func(epochs int)
}

// NewTrainingMonitor creates a monitor with default settings
func NewTrainingMonitor[T Numeric]() *TrainingMonitor[T] {
	return &TrainingMonitor[T]{
		width:          80,
		height:         20,
		output:         os.Stdout,
		updateInterval: 100 * time.Millisecond,
		bestMSE:        math.MaxFloat32,
		mseHistory:     make([]float32, 0, 1000),
		accuracyHistory: make([]float32, 0, 1000),
		learningRates:  make([]float32, 0, 1000),
	}
}

// Start begins monitoring
func (tm *TrainingMonitor[T]) Start(totalEpochs int) {
	tm.mu.Lock()
	tm.startTime = time.Now()
	tm.totalEpochs = totalEpochs
	tm.currentEpoch = 0
	tm.mu.Unlock()
	
	fmt.Fprintf(tm.output, "\n🚀 Training Monitor Started\n")
	fmt.Fprintf(tm.output, "   Total Epochs: %d\n\n", totalEpochs)
}

// Update records a training update
func (tm *TrainingMonitor[T]) Update(epoch int, mse float32, accuracy float32, lr float32) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	
	tm.currentEpoch = epoch
	tm.currentMSE = mse
	
	// Track history
	tm.mseHistory = append(tm.mseHistory, mse)
	tm.accuracyHistory = append(tm.accuracyHistory, accuracy)
	tm.learningRates = append(tm.learningRates, lr)
	
	// Check for improvement
	if mse < tm.bestMSE {
		oldBest := tm.bestMSE
		tm.bestMSE = mse
		if tm.onImprovement != nil {
			tm.onImprovement(oldBest, mse)
		}
	}
	
	// Check if we should update display
	now := time.Now()
	if now.Sub(tm.lastUpdate) >= tm.updateInterval {
		tm.lastUpdate = now
		tm.render()
	}
	
	// Trigger callbacks
	if tm.onEpoch != nil {
		tm.onEpoch(epoch, mse)
	}
}

// render displays the current state
func (tm *TrainingMonitor[T]) render() {
	// Clear previous output (simplified - works in most terminals)
	fmt.Fprintf(tm.output, "\033[2J\033[H")
	
	// Header
	elapsed := time.Since(tm.startTime)
	progress := float64(tm.currentEpoch) / float64(tm.totalEpochs)
	eta := time.Duration(0)
	if progress > 0 {
		eta = time.Duration(float64(elapsed) / progress) - elapsed
	}
	
	fmt.Fprintf(tm.output, "═══ Training Progress ═══════════════════════════════════════\n")
	fmt.Fprintf(tm.output, "Epoch: %d/%d (%.1f%%) | Time: %s | ETA: %s\n",
		tm.currentEpoch, tm.totalEpochs, progress*100, formatDuration(elapsed), formatDuration(eta))
	fmt.Fprintf(tm.output, "Current MSE: %.6f | Best MSE: %.6f\n", tm.currentMSE, tm.bestMSE)
	
	// Progress bar
	tm.renderProgressBar(progress)
	
	// MSE graph
	fmt.Fprintf(tm.output, "\n📊 MSE History:\n")
	tm.renderGraph(tm.mseHistory, 10)
	
	// Accuracy graph if available
	if len(tm.accuracyHistory) > 0 && tm.accuracyHistory[len(tm.accuracyHistory)-1] > 0 {
		fmt.Fprintf(tm.output, "\n📈 Accuracy History:\n")
		tm.renderGraph(tm.accuracyHistory, 10)
	}
	
	// Learning rate
	if len(tm.learningRates) > 0 {
		currentLR := tm.learningRates[len(tm.learningRates)-1]
		fmt.Fprintf(tm.output, "\n🎯 Learning Rate: %.6f\n", currentLR)
	}
}

// renderProgressBar draws a progress bar
func (tm *TrainingMonitor[T]) renderProgressBar(progress float64) {
	barWidth := tm.width - 20
	filled := int(progress * float64(barWidth))
	
	fmt.Fprintf(tm.output, "\n[")
	for i := 0; i < barWidth; i++ {
		if i < filled {
			fmt.Fprintf(tm.output, "█")
		} else {
			fmt.Fprintf(tm.output, "░")
		}
	}
	fmt.Fprintf(tm.output, "] %.1f%%\n", progress*100)
}

// renderGraph draws a simple ASCII graph
func (tm *TrainingMonitor[T]) renderGraph(data []float32, height int) {
	if len(data) == 0 {
		return
	}
	
	// Sample data if too many points
	samples := tm.width - 10
	step := 1
	if len(data) > samples {
		step = len(data) / samples
	}
	
	// Find min/max for scaling
	min, max := data[0], data[0]
	for i := 0; i < len(data); i += step {
		if data[i] < min {
			min = data[i]
		}
		if data[i] > max {
			max = data[i]
		}
	}
	
	// Prevent division by zero
	if max == min {
		max = min + 0.0001
	}
	
	// Draw graph
	for y := height - 1; y >= 0; y-- {
		fmt.Fprintf(tm.output, "│")
		threshold := min + (max-min)*float32(y)/float32(height-1)
		
		for x := 0; x < len(data); x += step {
			if data[x] >= threshold {
				fmt.Fprintf(tm.output, "▓")
			} else {
				fmt.Fprintf(tm.output, " ")
			}
		}
		
		if y == height-1 {
			fmt.Fprintf(tm.output, " %.4f", max)
		} else if y == 0 {
			fmt.Fprintf(tm.output, " %.4f", min)
		}
		fmt.Fprintf(tm.output, "\n")
	}
	
	// X-axis
	fmt.Fprintf(tm.output, "└")
	for i := 0; i < samples; i++ {
		fmt.Fprintf(tm.output, "─")
	}
	fmt.Fprintf(tm.output, "\n")
}

// Complete finishes monitoring and shows summary
func (tm *TrainingMonitor[T]) Complete() {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	
	elapsed := time.Since(tm.startTime)
	
	fmt.Fprintf(tm.output, "\n\n✅ Training Complete!\n")
	fmt.Fprintf(tm.output, "════════════════════════════════════════════════════════════\n")
	fmt.Fprintf(tm.output, "Total Time: %s\n", formatDuration(elapsed))
	fmt.Fprintf(tm.output, "Final MSE: %.6f\n", tm.currentMSE)
	fmt.Fprintf(tm.output, "Best MSE: %.6f\n", tm.bestMSE)
	
	if len(tm.accuracyHistory) > 0 {
		finalAcc := tm.accuracyHistory[len(tm.accuracyHistory)-1]
		fmt.Fprintf(tm.output, "Final Accuracy: %.2f%%\n", finalAcc*100)
	}
	
	// Calculate convergence rate
	if len(tm.mseHistory) > 10 {
		early := tm.mseHistory[10]
		final := tm.mseHistory[len(tm.mseHistory)-1]
		improvement := (early - final) / early * 100
		fmt.Fprintf(tm.output, "Improvement: %.1f%%\n", improvement)
	}
}

// ExpertMonitor tracks multiple experts training concurrently
type ExpertMonitor[T Numeric] struct {
	mu         sync.RWMutex
	experts    map[string]*ExpertStatus[T]
	startTime  time.Time
	output     io.Writer
	updateChan chan ExpertUpdate[T]
	stopChan   chan bool
}

// ExpertStatus tracks individual expert progress
type ExpertStatus[T Numeric] struct {
	Name         string
	Domain       string
	CurrentEpoch int
	CurrentMSE   float32
	Accuracy     float32
	Status       string // "training", "completed", "failed"
	StartTime    time.Time
	EndTime      time.Time
}

// ExpertUpdate represents a training update
type ExpertUpdate[T Numeric] struct {
	Name     string
	Epoch    int
	MSE      float32
	Accuracy float32
	Status   string
}

// NewExpertMonitor creates a monitor for multiple experts
func NewExpertMonitor[T Numeric]() *ExpertMonitor[T] {
	return &ExpertMonitor[T]{
		experts:    make(map[string]*ExpertStatus[T]),
		output:     os.Stdout,
		updateChan: make(chan ExpertUpdate[T], 100),
		stopChan:   make(chan bool),
	}
}

// Start begins monitoring multiple experts
func (em *ExpertMonitor[T]) Start(expertNames []string) {
	em.startTime = time.Now()
	
	// Initialize expert status
	for _, name := range expertNames {
		em.experts[name] = &ExpertStatus[T]{
			Name:      name,
			Status:    "waiting",
			StartTime: time.Now(),
		}
	}
	
	// Start update handler
	go em.handleUpdates()
	
	fmt.Fprintf(em.output, "\n🚀 Multi-Expert Training Monitor\n")
	fmt.Fprintf(em.output, "   Experts: %d\n\n", len(expertNames))
}

// Update sends an expert update
func (em *ExpertMonitor[T]) Update(update ExpertUpdate[T]) {
	em.updateChan <- update
}

// handleUpdates processes updates and renders
func (em *ExpertMonitor[T]) handleUpdates() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case update := <-em.updateChan:
			em.mu.Lock()
			if expert, ok := em.experts[update.Name]; ok {
				expert.CurrentEpoch = update.Epoch
				expert.CurrentMSE = update.MSE
				expert.Accuracy = update.Accuracy
				expert.Status = update.Status
				if update.Status == "completed" || update.Status == "failed" {
					expert.EndTime = time.Now()
				}
			}
			em.mu.Unlock()
			
		case <-ticker.C:
			em.render()
			
		case <-em.stopChan:
			return
		}
	}
}

// render displays all expert statuses
func (em *ExpertMonitor[T]) render() {
	em.mu.RLock()
	defer em.mu.RUnlock()
	
	// Clear screen
	fmt.Fprintf(em.output, "\033[2J\033[H")
	
	fmt.Fprintf(em.output, "═══ Multi-Expert Training Status ═══════════════════════════\n")
	fmt.Fprintf(em.output, "Elapsed: %s\n\n", formatDuration(time.Since(em.startTime)))
	
	// Table header
	fmt.Fprintf(em.output, "%-20s %-10s %-10s %-10s %-10s\n",
		"Expert", "Status", "Epoch", "MSE", "Accuracy")
	fmt.Fprintf(em.output, "%s\n", strings.Repeat("-", 70))
	
	// Expert rows
	completed := 0
	for _, expert := range em.experts {
		statusIcon := "⏳"
		switch expert.Status {
		case "training":
			statusIcon = "🔄"
		case "completed":
			statusIcon = "✅"
			completed++
		case "failed":
			statusIcon = "❌"
		}
		
		fmt.Fprintf(em.output, "%-20s %s %-8s %-10d %-10.6f %-10.2f%%\n",
			expert.Name,
			statusIcon,
			expert.Status,
			expert.CurrentEpoch,
			expert.CurrentMSE,
			expert.Accuracy*100)
	}
	
	// Summary
	fmt.Fprintf(em.output, "\n%s\n", strings.Repeat("-", 70))
	fmt.Fprintf(em.output, "Progress: %d/%d experts completed\n",
		completed, len(em.experts))
}

// Stop ends monitoring
func (em *ExpertMonitor[T]) Stop() {
	close(em.stopChan)
	em.render() // Final render
	
	fmt.Fprintf(em.output, "\n✅ All experts training complete!\n")
}

// Helper functions

func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	} else if d < time.Hour {
		return fmt.Sprintf("%.1fm", d.Minutes())
	}
	return fmt.Sprintf("%.1fh", d.Hours())
}