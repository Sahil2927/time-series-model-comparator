import matplotlib.pyplot as plt
import os

def plot_training_history(model_name, train_losses, val_losses):
    """
    Plots and saves the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_loss_curve.png')
    print(f"Plot saved to results/{model_name}_loss_curve.png")
    plt.close()