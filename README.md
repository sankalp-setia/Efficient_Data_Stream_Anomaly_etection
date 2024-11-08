# Real-Time Data Stream Anomaly Detection

This repository contains the code and documentation for a real-time anomaly detection system applied to continuous data streams. The system is designed to process high-velocity data efficiently and adapt to changes in the data distribution, making it suitable for applications in finance, cybersecurity, system health monitoring, and more.

## Project Overview

In the era of big data, monitoring and analyzing continuous data streams in real-time is essential for identifying irregularities or potential threats. This anomaly detection system uses a Z-Score-based adaptive sliding window method to detect anomalies in real time. It accommodates concept drift and seasonal variations and provides a real-time visualization of the data stream with highlighted anomalies.

## Features

- **Real-Time Anomaly Detection**: Uses Z-Score statistics and an adaptive sliding window approach.
- **Data Stream Simulation**: Generates synthetic data with trend, seasonality, noise, and anomalies for testing.
- **Real-Time Visualization**: Plots the data stream with anomalies highlighted, using Matplotlib.
- **Performance Metrics**: Evaluates model performance using a confusion matrix and precision-recall curve.

## Technologies Used

- **Python**: The core programming language used for this project.
- **Matplotlib**: For real-time visualization of the data stream and detected anomalies.
- **NumPy**: For numerical computations and handling data.
- **Scikit-Learn**: For evaluation metrics like confusion matrix and precision-recall curve.
