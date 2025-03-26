import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_with_uncertainty(model, data, device='cpu'):
    """
    Get predictions with uncertainty estimates

    Args:
        model: The trained model
        data: Input data
        device: Device to run inference on ('cpu' or 'cuda')

    Returns:
        mean_pred: Mean predictions
        std: Standard deviations (uncertainty)
    """
    model.eval()
    # Ensure data is on the correct device
    if isinstance(data, torch.Tensor):
        data = data.to(device)

    with torch.no_grad():
        mean_pred, std = model(data)

    return mean_pred.cpu().numpy(), std.cpu().numpy()

def plot_prediction_with_uncertainty(x_time, true_values, mean_pred, std, confidence=0.95):
    """
    Plot prediction with uncertainty bands
    """
    # Calculate z-score for the desired confidence interval
    z_score = {
        0.50: 0.674,
        0.68: 1.0,
        0.95: 1.96,
        0.99: 2.576
    }[confidence]

    plt.figure(figsize=(12, 6))
    plt.plot(x_time, true_values, 'k-', label='True Values')
    plt.plot(x_time, mean_pred, 'b-', label='Prediction')

    # Plot confidence intervals
    plt.fill_between(
        x_time,
        mean_pred - z_score * std,
        mean_pred + z_score * std,
        alpha=0.2,
        label=f'{int(confidence*100)}% Confidence Interval'
    )

    plt.legend()
    plt.title('Prediction with Uncertainty')
    plt.tight_layout()
    plt.show()

# Example usage:
# mean, std = predict_with_uncertainty(model, test_data)
# plot_prediction_with_uncertainty(time_array, true_values, mean, std)
