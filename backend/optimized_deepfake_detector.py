# backend/optimized_deepfake_detector.py
import torch
import torch.nn as nn
from transformers import Dinov2Model
# --- Updated/Added Imports ---
import cv2
import numpy as np
import logging
import os
from torchvision.transforms import functional as F # Use functional for transforms on tensors
from torchvision.transforms import Compose, Resize, Normalize, ToTensor # Import specific transforms
from typing import List # For type hinting
# --- End Updated/Added Imports ---

# Set up logging - basic config for standalone, main app may override
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deepfake_detector') # Specific logger name

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self, model_path="/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt"):
        """Initialize the fine-tuned DINOv2 model for deepfake detection on CPU."""
        super().__init__()
        self.device = torch.device("cpu") # Explicitly set to CPU
        logger.info(f"Initializing DINOv2 Deepfake Detector on {self.device}...")
        try:
            # Load base DINOv2 model
            self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-small")
            # Define the classifier head (use config for input size)
            classifier_input_size = self.dinov2.config.hidden_size # Typically 384 for dinov2-small
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2) # 2 classes: 0=Real, 1=Fake
            )

            # Load fine-tuned weights
            if not os.path.exists(model_path):
                 logger.error(f"Model weights file not found: {model_path}")
                 raise FileNotFoundError(f"Model weights not found: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)

            self.to(self.device)
            self.eval() # Set to evaluation mode

            # Define preprocessing (consistent for prediction and explanation)
            self.image_size = 224 # Standard for many ViTs, DINOv2 often uses this or larger
            self.preprocess = Compose([
                # Resize expects PIL image or Tensor C, H, W
                # Normalize expects Tensor C, H, W
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            # ToTensor() will be applied first to convert NumPy/PIL to Tensor

            logger.info(f"DINOv2 Deepfake Detector initialized successfully from {model_path}")
        except FileNotFoundError: # Re-raise specific error
             raise
        except Exception as e:
            logger.error(f"Failed to initialize deepfake detector: {str(e)}", exc_info=True)
            raise

    def forward(self, x, output_attentions=False): # Added output_attentions flag
        """Define the forward pass, optionally outputting attention weights."""
        # Pass the flag to the underlying DINOv2 model
        model_output = self.dinov2(x, output_attentions=output_attentions)
        # Use CLS token pooling (common ViT approach)
        features = model_output.last_hidden_state[:, 0]
        logits = self.classifier(features)

        if output_attentions:
            # Return logits AND the tuple of attention tensors
            return logits, model_output.attentions
        else:
            # Return only logits for standard prediction
            return logits

    @torch.no_grad()
    def process_batch(self, image_tensors: List[torch.Tensor]) -> List[float]:
        """
        Process a batch of preprocessed image tensors on the specified device.
        Assumes input tensors are already normalized and resized.
        Returns a list of deepfake scores (probability of being fake - class 1).
        """
        if not image_tensors:
            return []
        self.eval() # Ensure eval mode
        try:
            batch_tensor = torch.stack(image_tensors).to(self.device)
            outputs = self(batch_tensor) # Calls forward without output_attentions
            # Probability of class 1 (Fake)
            scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error processing preprocessed batch: {str(e)}", exc_info=True)
            # Return default score for each input tensor on error
            return [0.0] * len(image_tensors)

    @torch.no_grad()
    def predict_batch(self, frames: List[np.ndarray]) -> List[float]:
        """
        Predict deepfake scores for a batch of raw frames (BGR NumPy arrays).
        Handles ALL necessary preprocessing.
        """
        if not frames:
            return []
        self.eval() # Ensure eval mode
        preprocessed_tensors = []
        original_indices = [] # Keep track of which original frames were processed

        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                logger.warning(f"Skipping empty frame at index {i}.")
                continue
            try:
                # 1. Convert BGR to RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 2. Convert NumPy HWC to Tensor CHW and scale to [0, 1]
                img_tensor = ToTensor()(img_rgb)
                 # 3. Resize (expects CHW Tensor)
                img_resized = F.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
                # 4. Normalize (expects CHW Tensor)
                img_processed = self.preprocess(img_resized)
                preprocessed_tensors.append(img_processed)
                original_indices.append(i) # Store index of successfully processed frame
            except Exception as e:
                # Log error but continue to next frame
                logger.warning(f"Could not preprocess frame at index {i}, skipping: {e}")

        # Process only the successfully preprocessed tensors
        scores_processed = self.process_batch(preprocessed_tensors)

        # Map scores back to the original frame indices, filling failures with 0.0
        final_scores = [0.0] * len(frames)
        if len(scores_processed) == len(original_indices):
            for i, score in enumerate(scores_processed):
                 orig_idx = original_indices[i]
                 final_scores[orig_idx] = score
        else:
            # This case should ideally not happen if process_batch handles errors correctly
            logger.error(f"Score mapping failed: Processed scores ({len(scores_processed)}) != processed indices ({len(original_indices)}). Returning all zeros.")
            final_scores = [0.0] * len(frames)


        return final_scores

    # --- Explanation Methods ---

    @torch.no_grad()
    def get_attention_maps(self, frame: np.ndarray):
        """
        Generates attention map heatmap for a single frame (BGR NumPy array).
        Focuses on CLS token attention to patch tokens in the last layer.

        Args:
            frame (np.ndarray): Input frame (BGR format).

        Returns:
            np.ndarray | None: A normalized [0, 1] heatmap (H, W) or None on error.
        """
        if frame is None or frame.size == 0:
             logger.error("Cannot get attention maps for empty frame.")
             return None
        self.eval()
        logger.debug("Generating attention map...")
        try:
            # --- Preprocess single frame ---
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = ToTensor()(img_rgb) # CHW, [0, 1]
            img_resized = F.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
            img_processed = self.preprocess(img_resized) # Normalize
            input_batch = img_processed.unsqueeze(0).to(self.device) # Add batch dim, to device

            # --- Get Model Output with Attentions ---
            logits, attentions = self(input_batch, output_attentions=True)
            # attentions = tuple of attention tensors per layer (batch, heads, seq, seq)

            # --- Process Attention (Last Layer, Avg Heads, CLS to Patches) ---
            last_layer_attention = attentions[-1] # (1, num_heads, seq_len, seq_len)
            num_heads = last_layer_attention.shape[1]
            seq_len = last_layer_attention.shape[2] # Should be 1 (CLS) + num_patches
            num_patches = seq_len - 1

            # Average attention across heads
            avg_attention = torch.mean(last_layer_attention, dim=1).squeeze(0) # (seq_len, seq_len)

            # Attention from CLS token (idx 0) to patch tokens (idx 1 onwards)
            cls_to_patch_attention = avg_attention[0, 1:] # (num_patches,)

            # Reshape to 2D grid based on image size and patch size
            # Assuming square patches and input image
            patches_per_side = self.image_size // self.dinov2.config.patch_size # e.g. 224 // 16 = 14
            grid_size = patches_per_side
            if grid_size * grid_size != num_patches:
                logger.warning(f"Calculated grid size ({grid_size}x{grid_size}) doesn't match number of patches ({num_patches}). Check patch size config.")
                # Attempt to use calculated num_patches anyway if possible
                grid_size_from_patches = int(np.sqrt(num_patches))
                if grid_size_from_patches * grid_size_from_patches == num_patches:
                     grid_size = grid_size_from_patches
                     logger.warning(f"Using grid size {grid_size} based on patch count.")
                else:
                     logger.error("Cannot determine valid square grid size for attention map.")
                     return None

            attention_map = cls_to_patch_attention.reshape(grid_size, grid_size).cpu().numpy()

            # --- Upscale and Normalize Heatmap ---
            # Upscale small grid map to the input image size used by the model
            heatmap = cv2.resize(attention_map, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] range
            min_val, max_val = np.min(heatmap), np.max(heatmap)
            if max_val > min_val:
                 heatmap = (heatmap - min_val) / (max_val - min_val)
            else: # Handle flat attention map
                 heatmap = np.zeros_like(heatmap)

            logger.debug("Attention map generated successfully.")
            return heatmap # Return normalized heatmap (H, W)

        except Exception as e:
            logger.error(f"Error generating attention map: {e}", exc_info=True)
            return None

    def create_overlay(self, frame: np.ndarray, heatmap: np.ndarray, colormap=cv2.COLORMAP_JET, alpha=0.5) -> np.ndarray | None:
        """
        Overlays a normalized heatmap [0, 1] onto the original frame.

        Args:
            frame (np.ndarray): Original frame (BGR).
            heatmap (np.ndarray): Normalized heatmap [0, 1] (H, W - should match model input size e.g., 224x224).
            colormap (int): OpenCV colormap.
            alpha (float): Blending factor for the heatmap.

        Returns:
            np.ndarray | None: BGR image with overlay, resized to match heatmap, or None on error.
        """
        if heatmap is None:
            logger.warning("Cannot create overlay with None heatmap.")
            return None
        if frame is None or frame.size == 0:
            logger.error("Cannot create overlay with empty frame.")
            return None
        try:
            # Resize original frame to match heatmap size for blending
            frame_resized = cv2.resize(frame, (heatmap.shape[1], heatmap.shape[0]))

            # Apply colormap (needs 8-bit input)
            colored_heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)

            # Blend using cv2.addWeighted
            overlay = cv2.addWeighted(frame_resized, 1 - alpha, colored_heatmap, alpha, 0)

            return overlay
        except Exception as e:
             logger.error(f"Error creating overlay: {e}", exc_info=True)
             return None

# --- Standalone Test Section ---
if __name__ == "__main__":
    print("--- Running Standalone Detector Test ---") # Use print for standalone clarity

    # --- Configuration ---
    # !!! IMPORTANT: Verify these paths !!!
    MODEL_WEIGHTS_PATH = "/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt"
    TEST_IMAGE_PATH = "/Users/kartik/Desktop/vs/VeriStream/backend/test2.png" # MAKE SURE THIS IMAGE EXISTS

    # --- Initialization ---
    print(f"Initializing detector with weights from: {MODEL_WEIGHTS_PATH}")
    try:
        detector = OptimizedDeepfakeDetector(model_path=MODEL_WEIGHTS_PATH)
        print("Detector initialized successfully.")
    except Exception as init_err:
        print(f"FATAL: Failed to initialize detector: {init_err}")
        exit(1)

    # --- Load Test Image ---
    print(f"Loading test image from: {TEST_IMAGE_PATH}")
    if not os.path.exists(TEST_IMAGE_PATH):
         print(f"ERROR: Test image not found at specified path!")
         exit(1)

    test_frame = cv2.imread(TEST_IMAGE_PATH)
    if test_frame is None:
        print(f"ERROR: Failed to load test image using OpenCV.")
        exit(1)
    else:
         print(f"Test image loaded successfully (shape: {test_frame.shape})")

    # --- 1. Test Prediction Score ---
    print("\n--- Testing Prediction ---")
    try:
        scores = detector.predict_batch([test_frame])
        if scores:
            score = scores[0]
            print(f"Predicted Deepfake Score: {score:.4f}")
            print(f"Classification: {'FAKE' if score > 0.5 else 'REAL'} (threshold 0.5)")
        else:
            print("ERROR: Prediction returned no scores.")
    except Exception as pred_err:
        print(f"ERROR during prediction test: {pred_err}")

    # --- 2. Test Explanation Generation ---
    print("\n--- Testing Explanation (Attention Map) ---")
    try:
        heatmap = detector.get_attention_maps(test_frame)

        if heatmap is not None:
            print(f"Heatmap generated successfully (shape: {heatmap.shape}, range: [{heatmap.min():.2f}, {heatmap.max():.2f}]).")

            # --- 3. Test Overlay Creation ---
            print("Creating overlay image...")
            overlay_image = detector.create_overlay(test_frame, heatmap, alpha=0.6)

            if overlay_image is not None:
                # --- 4. Save Output ---
                output_filename = "attention_overlay_output.jpg"
                try:
                    success = cv2.imwrite(output_filename, overlay_image)
                    if success:
                        print(f"Successfully saved attention overlay to: {os.path.abspath(output_filename)}")
                    else:
                        print(f"ERROR: Failed to save overlay image (cv2.imwrite returned False).")
                except Exception as save_err:
                    print(f"ERROR: Failed to save overlay image to {output_filename}: {save_err}")
            else:
                print("ERROR: Overlay creation failed.")
        else:
            print("ERROR: Heatmap generation failed.")

    except Exception as explain_err:
        print(f"ERROR during explanation test: {explain_err}")

    print("\n--- Standalone Test Finished ---")