import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate toy time series data
np.random.seed(42)
time = np.arange(0, 100, 1)
signal = np.sin(0.1 * time) + np.random.normal(0, 0.1, size=len(time))

# Function to create time series features and labels for 48-ahead prediction with sliding window=2
def create_sequence_data(sequence, look_back, prediction_steps, step_size):
    X, y = [], []
    for i in range(0, len(sequence) - look_back - prediction_steps + 1, step_size):
        X.append(sequence[i : i + look_back])
        y.append(sequence[i + look_back : i + look_back + prediction_steps])
    return np.array(X), np.array(y)

# Define the look-back window size, prediction steps, and step size
look_back = 10
prediction_steps = 48
step_size = 2

# Create features and labels with sliding window
X, y = create_sequence_data(signal, look_back, prediction_steps, step_size)

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Initialize and train SVR model
svr_model = SVR()
svr_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = []
for i in range(len(X_test)):
    prediction_step = svr_model.predict(X_test[i:i + 1])
    predictions.append(prediction_step[0])

# Plot the results
plt.plot(time, signal, label="Actual Signal", color="blue")
plt.plot(
    time[look_back : len(X_train) + look_back : step_size],
    np.concatenate([y_train[:, 0], predictions]),
    label="SVR Predictions",
    color="red",
)
plt.legend()
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.title("SVR 48-Ahead Prediction with Sliding Window (Step Size = 2) on Toy Time Series Data")
plt.show()
