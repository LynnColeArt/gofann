package gofann

import (
	"math"
)

// Activation applies the activation function
func Activation[T Numeric](fn ActivationFunc, steepness, value T) T {
	switch fn {
	case Linear:
		return value
	case Threshold:
		if value > 0 {
			return T(1.0)
		}
		return T(0.0)
	case ThresholdSymmetric:
		if value > 0 {
			return T(1.0)
		}
		return T(-1.0)
	case Sigmoid:
		return T(1.0) / (T(1.0) + T(math.Exp(float64(-T(2.0)*steepness*value))))
	case SigmoidStepwise:
		v := T(2.0) * steepness * value
		if v <= -500 {
			return T(0.0)
		}
		if v >= 500 {
			return T(1.0)
		}
		return T(1.0) / (T(1.0) + T(math.Exp(float64(-v))))
	case SigmoidSymmetric:
		return T(2.0)/(T(1.0)+T(math.Exp(float64(-T(2.0)*steepness*value)))) - T(1.0)
	case SigmoidSymmetricStepwise:
		v := T(2.0) * steepness * value
		if v <= -500 {
			return T(-1.0)
		}
		if v >= 500 {
			return T(1.0)
		}
		return T(2.0)/(T(1.0)+T(math.Exp(float64(-v)))) - T(1.0)
	case Gaussian:
		v := float64(value * steepness)
		return T(math.Exp(-v * v))
	case GaussianSymmetric:
		v := float64(value * steepness)
		return T(math.Exp(-v*v)*2.0 - 1.0)
	case GaussianStepwise:
		v := value * steepness
		vSq := v * v
		if vSq > T(150) {
			if value > 0 {
				return T(0.0)
			} else {
				return T(1.0)
			}
		}
		return T(math.Exp(float64(-vSq)))
	case Elliot:
		return ((value * steepness) / (T(1.0) + abs(value*steepness))) * T(0.5) + T(0.5)
	case ElliotSymmetric:
		return (value * steepness) / (T(1.0) + abs(value*steepness))
	case LinearPiece:
		if value < -steepness {
			return T(0.0)
		} else if value > steepness {
			return T(1.0)
		}
		return value/steepness/T(2.0) + T(0.5)
	case LinearPieceSymmetric:
		if value < -steepness {
			return T(-1.0)
		} else if value > steepness {
			return T(1.0)
		}
		return value / steepness
	case Sin:
		return T(math.Sin(float64(value * steepness)))
	case Cos:
		return T(math.Cos(float64(value * steepness)))
	case SinSymmetric:
		return T(math.Sin(float64(value*steepness))) * T(2.0) - T(1.0)
	case CosSymmetric:
		return T(math.Cos(float64(value*steepness))) * T(2.0) - T(1.0)
	case LinearPieceRect:
		if value < 0 {
			return T(0.0)
		} else if value > T(1.0)/steepness {
			return T(1.0)
		}
		return value * steepness
	case LinearPieceRectLeaky:
		if value < 0 {
			return value * steepness * T(0.01)
		} else if value > T(1.0)/steepness {
			return T(1.0)
		}
		return value * steepness
	default:
		return value
	}
}

// ActivationDerivative computes the derivative of activation function
func ActivationDerivative[T Numeric](fn ActivationFunc, steepness, value, sum T) T {
	switch fn {
	case Linear:
		return T(1.0)
	case Threshold, ThresholdSymmetric:
		return T(0.0)
	case Sigmoid:
		// derivative: f'(x) = f(x) * (1 - f(x)) * 2 * steepness
		return T(2.0) * steepness * value * (T(1.0) - value)
	case SigmoidStepwise:
		// Same as sigmoid but with cutoff handling
		if value <= 0 || value >= 1 {
			return T(0.0)
		}
		return T(2.0) * steepness * value * (T(1.0) - value)
	case SigmoidSymmetric:
		// derivative: f'(x) = steepness * (1 - f(x)^2)
		return steepness * (T(1.0) - value*value)
	case SigmoidSymmetricStepwise:
		// Same as sigmoid symmetric but with cutoff
		if value <= -1 || value >= 1 {
			return T(0.0)
		}
		return steepness * (T(1.0) - value*value)
	case Gaussian:
		// derivative: f'(x) = -2 * x * steepness^2 * f(x)
		return T(-2.0) * sum * steepness * steepness * value
	case GaussianSymmetric:
		// derivative: f'(x) = -2 * x * steepness^2 * (f(x) + 1)
		return T(-2.0) * sum * steepness * steepness * (value + T(1.0))
	case GaussianStepwise:
		if value <= 0 || value >= 1 {
			return T(0.0)
		}
		return T(-2.0) * sum * steepness * steepness * value
	case Elliot:
		s := T(1.0) + abs(sum*steepness)
		return steepness * T(0.5) / (s * s)
	case ElliotSymmetric:
		s := T(1.0) + abs(sum*steepness)
		return steepness / (s * s)
	case LinearPiece:
		if sum < -steepness || sum > steepness {
			return T(0.0)
		}
		return T(1.0) / (steepness * T(2.0))
	case LinearPieceSymmetric:
		if sum < -steepness || sum > steepness {
			return T(0.0)
		}
		return T(1.0) / steepness
	case Sin:
		return T(math.Cos(float64(sum*steepness))) * steepness
	case Cos:
		return T(-math.Sin(float64(sum*steepness))) * steepness
	case SinSymmetric:
		return T(math.Cos(float64(sum*steepness))) * steepness * T(2.0)
	case CosSymmetric:
		return T(-math.Sin(float64(sum*steepness))) * steepness * T(2.0)
	case LinearPieceRect:
		if sum < 0 || sum > T(1.0)/steepness {
			return T(0.0)
		}
		return steepness
	case LinearPieceRectLeaky:
		if sum < 0 {
			return steepness * T(0.01)
		} else if sum > T(1.0)/steepness {
			return T(0.0)
		}
		return steepness
	default:
		return T(0.0)
	}
}

// ActivationName returns the string name of an activation function
func ActivationName(fn ActivationFunc) string {
	names := []string{
		"Linear",
		"Threshold",
		"ThresholdSymmetric",
		"Sigmoid",
		"SigmoidStepwise",
		"SigmoidSymmetric",
		"SigmoidSymmetricStepwise",
		"Gaussian",
		"GaussianSymmetric",
		"GaussianStepwise",
		"Elliot",
		"ElliotSymmetric",
		"LinearPiece",
		"LinearPieceSymmetric",
		"SinSymmetric",
		"CosSymmetric",
		"Sin",
		"Cos",
		"LinearPieceRect",
		"LinearPieceRectLeaky",
	}
	if int(fn) >= 0 && int(fn) < len(names) {
		return names[fn]
	}
	return "Unknown"
}