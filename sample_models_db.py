"""Sample models database management."""

import os
from pathlib import Path
from typing import List, Dict


SAMPLE_MODELS = {
    'iris_classifier.tflite': {
        'name': '🌸 Iris Classifier',
        'description': 'Classifies iris flower species',
        'task': 'Multi-class Classification',
        'input': '4 features (sepal/petal dimensions)',
        'output': '3 classes (Setosa, Versicolor, Virginica)',
        'use_case': 'Botany IoT, Plant monitoring',
        'complexity': 'Low',
        'expected_ram_kb': 32
    },
    'housing_regression.tflite': {
        'name': '🏠 Housing Price Predictor',
        'description': 'Predicts house prices from features',
        'task': 'Linear Regression',
        'input': '5 features (size, bedrooms, bathrooms, age, location)',
        'output': '1 continuous value (price)',
        'use_case': 'Real estate, Property valuation',
        'complexity': 'Low',
        'expected_ram_kb': 32
    },
    'digit_recognizer.tflite': {
        'name': '🔢 MNIST Digit Recognizer',
        'description': 'Recognizes handwritten digits 0-9',
        'task': 'Image Classification',
        'input': '784 pixels (28x28 image)',
        'output': '10 classes (digits 0-9)',
        'use_case': 'Document scanning, Postal automation',
        'complexity': 'Medium',
        'expected_ram_kb': 64
    },
    'activity_classifier.tflite': {
        'name': '🚶 Activity Recognition',
        'description': 'Recognizes human physical activity',
        'task': 'Time-series Classification',
        'input': '6 sensors (accel X/Y/Z, gyro X/Y/Z)',
        'output': '4 classes (Walking, Running, Sitting, Standing)',
        'use_case': 'Fitness trackers, Smart watches',
        'complexity': 'Medium',
        'expected_ram_kb': 48
    },
    'anomaly_detector.tflite': {
        'name': '⚠️ Sensor Anomaly Detector',
        'description': 'Detects anomalies in sensor readings',
        'task': 'Binary Classification',
        'input': '8 sensors (temperature, humidity, pressure, etc)',
        'output': '2 classes (Normal, Anomaly)',
        'use_case': 'IoT monitoring, Predictive maintenance',
        'complexity': 'High',
        'expected_ram_kb': 64
    }
}


def get_sample_database_path() -> Path:
    """Get path to sample_database folder."""
    return Path(__file__).parent / "sample_database"


def get_available_samples() -> List[Dict]:
    """Get list of available sample models with metadata.
    
    Returns:
        List of dicts with model info and path
    """
    sample_dir = get_sample_database_path()
    available = []
    
    for filename, metadata in SAMPLE_MODELS.items():
        model_path = sample_dir / filename
        if model_path.exists():
            available.append({
                'filename': filename,
                'path': str(model_path),
                'exists': True,
                **metadata
            })
        else:
            available.append({
                'filename': filename,
                'path': str(model_path),
                'exists': False,
                **metadata
            })
    
    return available


def get_sample_model_path(filename: str) -> str:
    """Get full path to a sample model by filename.
    
    Args:
        filename: Name of the model file
        
    Returns:
        Full path to the model
    """
    return str(get_sample_database_path() / filename)


def all_samples_available() -> bool:
    """Check if all sample models are available.
    
    Returns:
        True if all models exist
    """
    samples = get_available_samples()
    return all(s['exists'] for s in samples)
