# segmentation_model.py
from mmengine.config import Config
from mmseg.apis import init_model
import torch
import numpy as np
from PIL import Image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_ccnet_model(config_path, checkpoint_path, device='cpu'):
    """
    Load the CCNet segmentation model using updated MMSeg APIs
    """
    try:
        logger.info(f"Loading model from {config_path} with checkpoint {checkpoint_path}")
        
        # Check if config and checkpoint files exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load the configuration
        cfg = Config.fromfile(config_path)
        logger.debug("Config loaded successfully")
        
        # Update the config for inference
        cfg.model.pretrained = None
        
        # Initialize the model
        model = init_model(
            cfg,
            checkpoint_path,
            device=device
        )
        logger.info(f"Model initialized successfully on device: {device}")
        
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None

def preprocess_image(image, size=(512, 1024)):
    """
    Preprocess the image according to the test pipeline
    """
    try:
        logger.debug("Starting image preprocessing")
        
        # Resize image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            logger.debug("Converting RGBA to RGB")
            image = image.convert('RGB')
        
        # Resize image
        logger.debug(f"Resizing image to {size}")
        image = image.resize(size, Image.Resampling.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(image)
        logger.debug(f"Image array shape after conversion: {img_array.shape}")
        
        # Ensure RGB format
        if len(img_array.shape) == 2:
            logger.debug("Converting grayscale to RGB")
            img_array = np.stack([img_array] * 3, axis=-1)
            
        # Transpose image to (C, H, W) format
        img_array = img_array.transpose(2, 0, 1)
        logger.debug(f"Image array shape after transpose: {img_array.shape}")
        
        # Normalize image
        mean = np.array([123.675, 116.28, 103.53]).reshape(-1, 1, 1)
        std = np.array([58.395, 57.12, 57.375]).reshape(-1, 1, 1)
        img_array = (img_array - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0)
        logger.debug(f"Final tensor shape: {img_tensor.shape}")
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}", exc_info=True)
        return None

def run_inference(model_and_device, image):
    """
    Run inference on a single image
    """
    try:
        model, device = model_and_device
        logger.info("Starting inference")
        
        if model is None:
            raise ValueError("Model not initialized")
            
        # Preprocess image
        input_tensor = preprocess_image(image)
        if input_tensor is None:
            raise ValueError("Failed to preprocess image")
        
        # Move to correct device
        input_tensor = input_tensor.to(device)
        logger.debug(f"Input tensor moved to device {device}")
        
        # Run inference
        with torch.no_grad():
            # Create input dict for model
            data = dict(
                inputs=input_tensor,
                data_samples=None
            )
            
            # Run model inference
            result = model.test_step(data)
            logger.debug("Model inference completed")
            
            # Extract segmentation mask
            if hasattr(result, 'pred_sem_seg'):
                seg_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
            else:
                seg_mask = result[0].pred_sem_seg.data.cpu().numpy().squeeze()
            
            logger.debug(f"Segmentation mask shape: {seg_mask.shape}")
            return seg_mask
            
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        return None

# Initialize model with proper error handling
def initialize_model():
    try:
        # Update these paths to match your actual file locations
        config_file = '../config/ccnet_r101-d8_512x1024_80k.py'
        checkpoint_file = '../checkpoints/iter_80000.pth'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing model on device: {device}")
        model, device = load_ccnet_model(config_file, checkpoint_file, device)
        
        if model is None:
            raise ValueError("Failed to load model")
        
        logger.info(f"Model initialized successfully on {device}")
        return (model, device)
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        return None

# Global model instance and device
segmentation_model = initialize_model()