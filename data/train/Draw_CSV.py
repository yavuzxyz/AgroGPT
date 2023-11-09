import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("C:/Users/yavuz/OneDrive - uludag.edu.tr/2-YOLOv8/Snail_segment/train/results.csv")

# Display the first few rows of the data
print(data.head())

# Strip leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Display the first few rows of the data
print(data.head())

# Now let's try to plot the data again
# Change default font
plt.rcParams["font.family"] = "Times New Roman"

# Plotting training loss metrics
plt.figure(figsize=(11,8))
plt.plot(data['epoch'], data['train/box_loss'], label='Box Loss')
plt.plot(data['epoch'], data['train/cls_loss'], label='Cls Loss')
plt.plot(data['epoch'], data['train/dfl_loss'], label='Dfl Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plotting validation loss metrics
plt.figure(figsize=(11,8))
plt.plot(data['epoch'], data['val/box_loss'], label='Box Loss')
plt.plot(data['epoch'], data['val/cls_loss'], label='Cls Loss')
plt.plot(data['epoch'], data['val/dfl_loss'], label='Dfl Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plotting metrics
plt.figure(figsize=(11,8))
plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')
plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP50')
plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP50-95')
plt.xlabel('Epoch')
plt.ylabel('Metric Score')
plt.title('Metrics over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plotting learning rates
plt.figure(figsize=(11,8))
plt.plot(data['epoch'], data['lr/pg0'], label='Learning Rate pg0')
plt.plot(data['epoch'], data['lr/pg1'], label='Learning Rate pg1')
plt.plot(data['epoch'], data['lr/pg2'], label='Learning Rate pg2')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Epochs')
plt.legend()
plt.grid(True)
plt.show()
