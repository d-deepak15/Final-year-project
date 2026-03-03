"""Model conversion utilities for converting various ML formats to TFLite."""

import os
import tempfile
import shutil
from pathlib import Path
import tensorflow as tf
from typing import Tuple, Optional


class ModelConverter:
    """Handles conversion of ML models to TensorFlow Lite format."""
    
    SUPPORTED_FORMATS = {
        '.h5': 'Keras HDF5',
        '.pb': 'TensorFlow SavedModel (pb)',
        '.tflite': 'TensorFlow Lite (no conversion needed)',
        '.savedmodel': 'TensorFlow SavedModel directory'
    }
    
    @staticmethod
    def get_supported_extensions() -> list:
        """Return list of supported file extensions."""
        return list(ModelConverter.SUPPORTED_FORMATS.keys())
    
    @staticmethod
    def convert_keras_to_tflite(model_path: str, output_path: str) -> Tuple[bool, str]:
        """Convert Keras model (.h5) to TFLite.
        
        Args:
            model_path: Path to .h5 Keras model
            output_path: Path to save converted .tflite model
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load Keras model
            model = tf.keras.models.load_model(model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save converted model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return True, f"✅ Successfully converted Keras model to TFLite"
        except Exception as e:
            return False, f"❌ Keras conversion failed: {str(e)}"
    
    @staticmethod
    def convert_savedmodel_to_tflite(model_path: str, output_path: str) -> Tuple[bool, str]:
        """Convert TensorFlow SavedModel to TFLite.
        
        Args:
            model_path: Path to SavedModel directory
            output_path: Path to save converted .tflite model
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save converted model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return True, f"✅ Successfully converted SavedModel to TFLite"
        except Exception as e:
            return False, f"❌ SavedModel conversion failed: {str(e)}"
    
    @staticmethod
    def convert_pb_to_tflite(model_path: str, output_path: str) -> Tuple[bool, str]:
        """Convert TensorFlow frozen graph (.pb) to TFLite.
        
        Args:
            model_path: Path to .pb frozen graph
            output_path: Path to save converted .tflite model
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Load frozen graph
            converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file=model_path,
                input_arrays=['input'],
                output_arrays=['output']
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save converted model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return True, f"✅ Successfully converted .pb to TFLite"
        except Exception as e:
            return False, f"❌ .pb conversion failed: {str(e)}"
    
    @staticmethod
    def validate_tflite(model_path: str) -> Tuple[bool, str]:
        """Validate if a file is a valid TFLite model.
        
        Args:
            model_path: Path to TFLite model
            
        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            msg = f"✅ Valid TFLite model\n"
            msg += f"   Inputs: {len(input_details)} | Outputs: {len(output_details)}"
            return True, msg
        except Exception as e:
            return False, f"❌ Invalid TFLite model: {str(e)}"
    
    @staticmethod
    def process_model_file(uploaded_file, output_dir: str) -> Tuple[bool, str, Optional[str]]:
        """Process uploaded file and convert to TFLite if needed.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            output_dir: Directory to save converted model
            
        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get file extension and name
            file_name = uploaded_file.name
            file_ext = Path(file_name).suffix.lower()
            
            # Save uploaded file temporarily
            temp_input_path = os.path.join(output_dir, file_name)
            with open(temp_input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # If already TFLite, just validate and return
            if file_ext == '.tflite':
                is_valid, validation_msg = ModelConverter.validate_tflite(temp_input_path)
                if is_valid:
                    final_path = temp_input_path
                    return True, validation_msg, final_path
                else:
                    os.remove(temp_input_path)
                    return False, validation_msg, None
            
            # Convert to TFLite
            output_tflite = os.path.join(output_dir, f"model_converted.tflite")
            
            if file_ext == '.h5':
                success, msg = ModelConverter.convert_keras_to_tflite(temp_input_path, output_tflite)
            elif file_ext == '.pb':
                success, msg = ModelConverter.convert_pb_to_tflite(temp_input_path, output_tflite)
            elif file_ext == '.savedmodel' or os.path.isdir(temp_input_path):
                success, msg = ModelConverter.convert_savedmodel_to_tflite(temp_input_path, output_tflite)
            else:
                return False, f"❌ Unsupported format: {file_ext}", None
            
            if not success:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                return False, msg, None
            
            # Clean up original file
            if os.path.exists(temp_input_path) and temp_input_path != output_tflite:
                os.remove(temp_input_path)
            
            # Validate converted model
            is_valid, validation_msg = ModelConverter.validate_tflite(output_tflite)
            if is_valid:
                return True, msg + "\n" + validation_msg, output_tflite
            else:
                if os.path.exists(output_tflite):
                    os.remove(output_tflite)
                return False, validation_msg, None
                
        except Exception as e:
            return False, f"❌ Processing failed: {str(e)}", None


def get_models_temp_dir() -> str:
    """Get or create the temporary models directory.
    
    Returns:
        Path to temporary models directory
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'tinyml_profiler_models')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def cleanup_models_dir(models_dir: str) -> bool:
    """Delete all models in the temporary directory.
    
    Args:
        models_dir: Path to models directory
        
    Returns:
        True if cleanup was successful
    """
    try:
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)
            return True
        return False
    except Exception as e:
        print(f"Cleanup error: {e}")
        return False
