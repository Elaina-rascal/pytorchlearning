# visual.py
import matplotlib.pyplot as plt

def plot_loss(losses, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="Average Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()