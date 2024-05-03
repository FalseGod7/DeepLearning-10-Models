import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import BernoulliRBM

# Load your CSV file containing reviews and sentiments
# Replace 'your_dataset.csv' with the actual path to your dataset
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1 = pd.read_csv('Pos_Neg_Final.csv')

# Assuming your CSV has two columns: 'Summary' and 'Sentiment' (positive, negative, neutral)
reviews = df['Summary_Tamil'].values
sentiments = df1['Sentiment_1'].values

# Convert non-string values to strings (if any)
reviews = [str(review) for review in reviews]

# Tokenize the text and pad sequences
vocab_size = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
x = pad_sequences(sequences)

# Convert sentiments to one-hot encoded labels
num_classes = 3
y = tf.keras.utils.to_categorical(sentiments, num_classes=num_classes)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# RBM model
rbm = BernoulliRBM(n_components=256, learning_rate=0.1, batch_size=64, n_iter=20, verbose=1, random_state=42)

# Create an MLP model (without using Pipeline)
mlp_model = Sequential([
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the MLP model
mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the RBM separately (if needed)
rbm.fit(x_train_scaled)

# Get transformed features from RBM
x_train_rbm_features = rbm.transform(x_train_scaled)
x_test_rbm_features = rbm.transform(x_test_scaled)

# Train the MLP model on RBM features
mlp_model.fit(x_train_rbm_features, y_train)

# Generate predictions
y_pred = mlp_model.predict(x_test_rbm_features)

# Classification report
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_true, y_pred_classes))

# Calculate AUC for each class
auc_scores = []
for i in range(num_classes):
    auc_scores.append(roc_auc_score(y_test[:, i], y_pred[:, i]))
    print(f"AUC for class {i}: {auc_scores[i]:.4f}")

# Plot multiclass ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=3, label=f'ROC curve (class {i}, area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()
