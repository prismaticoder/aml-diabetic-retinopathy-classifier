import json
import matplotlib.pyplot as plt

# Change this path if needed
log_path = "logs/Lawrence Attoh_mlp_mixer_v2_addblock_train_128_0.0005.json"

with open(log_path, "r") as f:
    log = json.load(f)

epochs = [entry["epoch"] for entry in log["epoch_logs"]]
train_loss = [entry["train_loss"] for entry in log["epoch_logs"]]
val_loss = [entry["val_loss"] for entry in log["epoch_logs"]]
train_acc = [entry["train_acc"] for entry in log["epoch_logs"]]
val_acc = [entry["val_acc"] for entry in log["epoch_logs"]]
val_qwk = [entry["val_qwk"] for entry in log["epoch_logs"]]

# Plot Loss
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Plot QWK
plt.figure()
plt.plot(epochs, val_qwk, label="Validation QWK", color="purple")
plt.xlabel("Epoch")
plt.ylabel("Quadratic Weighted Kappa")
plt.title("Validation QWK Score")
plt.legend()
plt.grid(True)
plt.show()
