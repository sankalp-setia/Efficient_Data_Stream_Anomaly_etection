import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

def data_stream_generator(total_points=1000, anomaly_ratio=0.02, seed=None):
    """
    Simulates a real-time data stream with trend, seasonality, noise, and anomalies.

    Parameters:
    - total_points: Total number of data points to generate.
    - anomaly_ratio: Fraction of points that are anomalies.
    - seed: Random seed for reproducibility.

    Yields:
    - A tuple containing the next data point and a boolean indicating if it's an anomaly.
    """
    if seed is not None:
        np.random.seed(seed)
    
    trend = 0.0
    seasonality_period = 50
    for i in range(total_points):
        trend += 0.01
        season = 10 * np.sin(2 * np.pi * i / seasonality_period)
        noise = np.random.normal(0, 1)
        value = trend + season + noise

        if np.random.random() < anomaly_ratio:
            anomaly = np.random.choice([np.random.uniform(15, 20), np.random.uniform(-20, -15)])
            value += anomaly
            yield value, True
        else:
            yield value, False

class AnomalyDetector:
    def __init__(self, window_size=100, z_threshold=3.0):
        """
        Initializes the anomaly detector with a sliding window.

        Parameters:
        - window_size: Number of recent data points to consider.
        - z_threshold: Z-score threshold for anomaly detection.
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.window = deque(maxlen=self.window_size)
        self.mean = 0.0
        self.std = 1.0

    def update_statistics(self, new_value):
        if len(self.window) == self.window_size:
            old = self.window.popleft()
            self.mean += (new_value - old) / self.window_size
            self.window.append(new_value)
            self.std = np.std(self.window)
        else:
            self.window.append(new_value)
            self.mean = np.mean(self.window)
            self.std = np.std(self.window)

    def is_anomaly(self, value):
        z_score = 0 if self.std == 0 else (value - self.mean) / self.std
        return abs(z_score) > self.z_threshold, z_score

class RealTimePlot:
    def __init__(self, max_points=1000):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.max_points = max_points
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.anomalies_x = deque(maxlen=max_points)
        self.anomalies_y = deque(maxlen=max_points)
        self.line_normal, = self.ax.plot([], [], label='Data')
        self.line_anomaly, = self.ax.plot([], [], 'ro', label='Anomaly')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.set_title('Real-Time Data Stream with Anomaly Detection')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()

    def update_plot(self, x, y, is_anomaly):
        self.x_data.append(x)
        self.y_data.append(y)
        if is_anomaly:
            self.anomalies_x.append(x)
            self.anomalies_y.append(y)
        
        self.line_normal.set_data(self.x_data, self.y_data)
        self.line_anomaly.set_data(self.anomalies_x, self.anomalies_y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

def main():
    TOTAL_POINTS = 1000
    WINDOW_SIZE = 100
    Z_THRESHOLD = 2.1
    ANOMALY_RATIO = 0.02  # 2% anomalies
    SLEEP_TIME = 0.01  # Seconds between data points

    detector = AnomalyDetector(window_size=WINDOW_SIZE, z_threshold=Z_THRESHOLD)
    plotter = RealTimePlot(max_points=TOTAL_POINTS)
    stream = data_stream_generator(total_points=TOTAL_POINTS, anomaly_ratio=ANOMALY_RATIO, seed=22)

    # Tracking variables
    detected_anomalies = 0
    actual_anomalies = 0
    false_positives = 0
    total_normal = 0

    # Lists for metrics
    y_true = []
    y_pred = []
    y_scores = []

    for idx, (value, actual_anomaly) in enumerate(stream):
        detector.update_statistics(value)
        detected_anomaly, z_score = detector.is_anomaly(value)
        plotter.update_plot(idx, value, detected_anomaly)
        
        # Collect true labels and predictions
        y_true.append(actual_anomaly)
        y_pred.append(detected_anomaly)
        y_scores.append(abs(z_score))
        
        if actual_anomaly:
            actual_anomalies += 1
            if detected_anomaly:
                detected_anomalies += 1
                print(f"Anomaly detected at index {idx}: Value={value:.2f}, Z-Score={z_score:.2f}")
        else:
            total_normal += 1
            if detected_anomaly:
                false_positives += 1
                print(f"False Positive at index {idx}: Value={value:.2f}, Z-Score={z_score:.2f}")
        
        time.sleep(SLEEP_TIME)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    
    # Calculate Detection Rate and False Positive Rate
    detection_rate = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    false_positive_rate = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    
    print(f"Detection Rate: {detection_rate:.2f}%")
    print(f"False Positive Rate: {false_positive_rate:.2f}%")
    
    # Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, cmap='Blues')

    # Show all ticks and label them
    ax_cm.set_xticks(np.arange(2))
    ax_cm.set_yticks(np.arange(2))
    ax_cm.set_xticklabels(['Normal', 'Anomaly'])
    ax_cm.set_yticklabels(['Normal', 'Anomaly'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(2):
        for j in range(2):
            text = ax_cm.text(j, i, cm[i, j],
                           ha="center", va="center", color="black")

    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig('confusion_matrix.png')
    plt.show()

    # Plot Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    plt.show()

    # Save Real-Time Plot as Image
    plotter.fig.savefig('real_time_plot.png')
    print("\nReal-time plot saved as 'real_time_plot.png'.")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()