"""
Generate 5 sample TFLite models for different ML tasks on IoT devices.
Run this script once to create the sample models in sample_database folder.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path


def create_sample_models_directory():
    """Create sample_database directory if it doesn't exist."""
    sample_dir = Path(__file__).parent / "sample_database"
    sample_dir.mkdir(exist_ok=True)
    return sample_dir


def create_iris_classifier(output_path):
    """Iris Flower Classification (3-class classification)
    
    Predicts flower species: Setosa, Versicolor, Virginica
    Input: 4 features (sepal length, sepal width, petal length, petal width)
    """
    print("Creating Iris Classifier...")
    
    # Generate synthetic Iris-like data
    np.random.seed(42)
    X_train = np.random.rand(150, 4).astype(np.float32) * [8, 4.5, 7, 2.5]
    y_train = np.array([0]*50 + [1]*50 + [2]*50)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Iris Classifier saved: {output_path}")


def create_regression_model(output_path):
    """Housing Price Prediction (Linear Regression)
    
    Predicts house price based on features
    Input: 5 features (size, bedrooms, bathrooms, age, location_score)
    """
    print("Creating Housing Price Predictor...")
    
    # Generate synthetic housing data
    np.random.seed(42)
    X_train = np.random.rand(200, 5).astype(np.float32) * [5000, 5, 3, 50, 10]
    y_train = (X_train[:, 0] * 0.1 + X_train[:, 1] * 50000 + 
               X_train[:, 2] * 30000 + np.random.rand(200) * 50000).astype(np.float32)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # Linear output for regression
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Housing Predictor saved: {output_path}")


def create_digit_recognizer(output_path):
    """Digit Recognition (MNIST-like, Binary/Multi-class)
    
    Recognizes handwritten digits 0-9
    Input: 28x28 pixel image (flattened to 784)
    """
    print("Creating Digit Recognizer...")
    
    # Generate synthetic digit-like data
    np.random.seed(42)
    X_train = np.random.rand(1000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 1000)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Digit Recognizer saved: {output_path}")


def create_activity_classifier(output_path):
    """Activity Recognition (Time-series Classification)
    
    Recognizes human activity from sensor data
    Input: 6 features from accelerometer/gyroscope (x, y, z for 2 sensors)
    Classes: Walking, Running, Sitting, Standing
    """
    print("Creating Activity Classifier...")
    
    # Generate synthetic sensor data
    np.random.seed(42)
    X_train = np.random.rand(400, 6).astype(np.float32) * 10
    y_train = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Activity Classifier saved: {output_path}")


def create_anomaly_detector(output_path):
    """Sensor Anomaly Detection (Binary Classification)
    
    Detects anomalies in sensor readings
    Input: 8 features from various sensors (temperature, humidity, pressure, etc)
    Classes: Normal (0), Anomaly (1)
    """
    print("Creating Anomaly Detector...")
    
    # Generate synthetic sensor data
    np.random.seed(42)
    X_train = np.random.rand(500, 8).astype(np.float32) * [50, 100, 1000, 50, 100, 50, 100, 50]
    # Add some anomalies
    anomalies = np.random.rand(100, 8).astype(np.float32) * [200, 200, 2000, 200, 200, 200, 200, 200]
    X_train = np.vstack([X_train, anomalies])
    y_train = np.array([0]*500 + [1]*100)
    
    # Shuffle
    indices = np.random.permutation(len(X_train))
    X_train, y_train = X_train[indices], y_train[indices]
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=0)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Anomaly Detector saved: {output_path}")


def generate_all_samples():
    """Generate all 5 sample models."""
    sample_dir = create_sample_models_directory()
    
    print(f"\n{'='*60}")
    print(f"  Generating 5 Sample TFLite Models for IoT")
    print(f"  Output directory: {sample_dir}")
    print(f"{'='*60}\n")
    
    models = [
        ('iris_classifier.tflite', create_iris_classifier, 'Iris Flower Classification'),
        ('housing_regression.tflite', create_regression_model, 'Housing Price Prediction'),
        ('digit_recognizer.tflite', create_digit_recognizer, 'MNIST Digit Recognition'),
        ('activity_classifier.tflite', create_activity_classifier, 'Activity Recognition'),
        ('anomaly_detector.tflite', create_anomaly_detector, 'Sensor Anomaly Detection'),
    ]
    
    for filename, creator_func, description in models:
        output_path = sample_dir / filename
        creator_func(str(output_path))
    
    print(f"\n{'='*60}")
    print(f"✅ All sample models generated successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_all_samples()
