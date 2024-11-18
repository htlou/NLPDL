import matplotlib.pyplot as plt

# Data from the log
epochs = list(range(1, 51))
train_loss = [
    5.907, 5.010, 4.254, 3.657, 3.204, 2.862, 2.606, 2.406, 2.250, 2.133, 2.017, 1.924, 
    1.840, 1.767, 1.703, 1.647, 1.582, 1.535, 1.490, 1.458, 1.415, 1.376, 1.351, 1.315, 
    1.287, 1.259, 1.234, 1.206, 1.188, 1.172, 1.140, 1.120, 1.108, 1.086, 1.073, 1.057, 
    1.036, 1.026, 1.011, 1.001, 0.981, 0.978, 0.959, 0.955, 0.940, 0.930, 0.921, 0.909, 
    0.902, 0.895
]
val_loss = [
    5.523, 4.887, 4.472, 4.255, 4.140, 4.070, 4.042, 4.020, 4.026, 4.030, 4.066, 4.066, 
    4.085, 4.123, 4.145, 4.161, 4.195, 4.226, 4.247, 4.276, 4.321, 4.332, 4.363, 4.395, 
    4.443, 4.448, 4.492, 4.524, 4.510, 4.563, 4.587, 4.605, 4.639, 4.670, 4.692, 4.680, 
    4.711, 4.767, 4.759, 4.826, 4.842, 4.882, 4.885, 4.903, 4.921, 4.969, 4.989, 4.966, 
    4.989, 5.020
]
bleu = [
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 
    0.09, 0.09, 0.10, 0.10, 0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 
    0.12, 0.12, 0.13, 0.13, 0.13, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 
    0.15, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.16, 0.16, 0.16, 0.15, 
    0.15, 0.16
]
perplexity = [
    250.38, 132.55, 87.55, 70.48, 62.77, 58.58, 56.92, 55.69, 56.02, 56.26, 58.33, 
    58.31, 59.43, 61.75, 63.12, 64.15, 66.33, 68.47, 69.87, 71.98, 75.26, 76.10, 
    78.47, 81.02, 85.01, 85.49, 89.29, 92.19, 90.89, 95.87, 98.24, 99.99, 103.45, 
    106.69, 109.02, 107.77, 111.13, 117.61, 116.61, 124.74, 126.78, 131.84, 132.32, 
    134.65, 137.07, 143.83, 146.86, 143.45, 146.82, 151.45
]

# Plotting the graphs
plt.figure(figsize=(12, 8))

# Plot train and validation loss
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()

# Plot BLEU score
plt.subplot(2, 2, 2)
plt.plot(epochs, bleu, label="BLEU Score", color='green')
plt.xlabel("Epochs")
plt.ylabel("BLEU Score")
plt.title("Validation BLEU Score")

# Plot Perplexity
plt.subplot(2, 2, 3)
plt.plot(epochs, perplexity, label="Perplexity", color='red')
plt.xlabel("Epochs")
plt.ylabel("Perplexity")
plt.title("Validation Perplexity")

plt.tight_layout()
plt.savefig('images/train.png')