# Temporal Uncertainty Modeling in MASTER

This document explains how uncertainty modeling works in the MASTER model, where it happens in the code, and how to use it.

## What is Uncertainty Modeling?

Traditional time series forecasting models only predict point estimates (mean values) without indicating how confident the model is in its predictions. Uncertainty modeling addresses this by outputting:

1. **Mean prediction**: The expected value (like traditional models)
2. **Uncertainty estimate**: How confident the model is in that prediction

This is especially valuable for financial time series where risk assessment is critical.

## Implementation in the MASTER Model

### Key Components and Where They Appear

#### 1. Model Architecture (`master.py`)

- **TemporalAttention**: Modified to output both mean embeddings and uncertainty embeddings
  ```python
  # In TemporalAttention.forward():
  h_uncertainty = self.uncertainty_trans(z)  # [N, T, D]
  uncertainty_output = torch.matmul(lam, h_uncertainty).squeeze(1)  # [N, 1, D]
  return output, uncertainty_output
  ```

- **MASTER**: Has separate decoders for mean and uncertainty
  ```python
  # In MASTER.__init__():
  self.mean_decoder = nn.Linear(d_model, 1)
  self.uncertainty_decoder = nn.Linear(d_model, 1)

  # In MASTER.forward():
  mean_pred = self.mean_decoder(mean_embedding).squeeze(-1)
  log_var = self.uncertainty_decoder(uncertainty_embedding).squeeze(-1)
  return mean_pred, log_var
  ```

- **MASTERModel**: Handles different outputs for training vs inference
  ```python
  # In MASTERModel.forward():
  if self.training:
      return mean_pred, log_var
  else:
      return mean_pred, torch.exp(0.5 * log_var)  # convert log_var to std
  ```

#### 2. Loss Function (`loss.py`)

The `GaussianNLLLoss` calculates loss based on both prediction accuracy and uncertainty calibration:

```python
# In GaussianNLLLoss.forward():
var = torch.exp(log_var) + self.eps
loss = 0.5 * (log_var + ((target - mean) ** 2) / var)
```

This loss function:
- Penalizes high variance predictions (via the log_var term)
- But also allows higher variance when errors are large (via the squared error divided by variance)
- Automatically calibrates uncertainty estimates during training

#### 3. Integration in Base Model (`base_model.py`)

- **loss_fn**: Detects when model outputs uncertainty and uses appropriate loss
- **predict**: Processes and returns uncertainty alongside predictions
- **fit**: Handles the case where predict returns 3 values (predictions, metrics, uncertainty)

#### 4. Analysis and Visualization (`inference_example.py`)

- **predict_with_uncertainty**: Returns both mean predictions and uncertainties
- **plot_prediction_with_uncertainty**: Visualizes predictions with confidence intervals

## How It Affects Performance

### 1. Training Dynamics

- The model is trained to balance prediction accuracy and uncertainty calibration
- When the model is uncertain about a prediction, it can express this through higher variance
- This mitigates the impact of unpredictable data points on model training

### 2. Inference Benefits

- **Risk Assessment**: Identifies which predictions have higher uncertainty
- **Confidence Intervals**: Creates probabilistic bounds around predictions
- **Decision Support**: More informed trading decisions based on both predictions and confidence

### 3. Performance Metrics

While traditional metrics (IC, ICIR, etc.) are still calculated on mean predictions, additional benefits include:
- More robust models through better uncertainty handling
- Ability to filter out high-uncertainty predictions
- Possibility of weighted decision-making based on confidence levels

## How to Use Uncertainty Estimates

### 1. During Training

The model automatically uses uncertainty modeling during training with the GaussianNLLLoss.

### 2. During Inference

```python
# Get predictions with uncertainty
result = model.predict(test_data)
if isinstance(result, tuple) and len(result) == 3:
    predictions, metrics, uncertainty = result

    # Create confidence intervals (95%)
    lower_bound = predictions - 1.96 * uncertainty
    upper_bound = predictions + 1.96 * uncertainty
```

### 3. Visualizing Uncertainty

```python
from inference_example import plot_prediction_with_uncertainty

plot_prediction_with_uncertainty(
    time_array,      # x-axis values
    true_values,     # actual values
    predictions,     # mean predictions
    uncertainty,     # standard deviations
    confidence=0.95  # confidence level
)
```

## Integration in Your Pipeline

To leverage uncertainty in a trading pipeline:

1. **Risk Management**: Scale position sizes inversely with uncertainty
2. **Filtering**: Set minimum confidence thresholds before taking positions
3. **Portfolio Allocation**: Allocate more capital to high-confidence predictions

```python
# Example of risk-adjusted decision making
def risk_adjusted_signals(predictions, uncertainty):
    # Scale signals by inverse of uncertainty
    confidence_weight = 1.0 / (1.0 + uncertainty)
    adjusted_signals = predictions * confidence_weight
    return adjusted_signals
```

## Mathematical Basis

The model outputs are based on a Gaussian probability distribution where:
- The mean (μ) represents the expected prediction
- The variance (σ²) represents the uncertainty

The probability density function is:
p(y|x) = (1/√(2πσ²)) * exp(-(y-μ)²/(2σ²))

Where:
- y is the actual value
- x is the input features
- μ is the predicted mean
- σ² is the predicted variance

## Conclusion

Uncertainty modeling transforms MASTER from a deterministic forecasting model into a probabilistic one, providing not just what might happen, but how confident the model is in that assessment. This is particularly valuable in financial markets where understanding risk is as important as seeking returns.
