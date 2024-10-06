#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pickle
import json

#%%
with open ('test_pred.pkl', 'rb') as f:
    true_labels, predicted_labels = pickle.load(f)
# # Sample data (replace these with your actual labels)
# true_labels = np.random.choice(['label1', 'label2', 'label3', 'label4', 'label5'], size=100)
# predicted_labels = np.random.choice(['label1', 'label2', 'label3', 'label4', 'label5'], size=100)
with open ('label_int_mapping_TrInF.json', 'rb') as f:
    label_int_mapping = json.load(f)
int_label_mapping = {v: k for k, v in label_int_mapping.items()}

true_labels = [int_label_mapping[i] for i in true_labels]
predicted_labels = [int_label_mapping[i] for i in predicted_labels]
# Create confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues', 
            xticklabels=np.unique(true_labels), 
            yticklabels=np.unique(true_labels),
            linewidths=0.5, linecolor='black')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.xticks(rotation=90, ha='right')  # Rotate labels for better readability
plt.yticks(rotation=0)  # Keep y-labels horizontal
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('plot.png')