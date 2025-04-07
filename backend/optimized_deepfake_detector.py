# backend/optimized_deepfake_detector.py
import torch
import torch.nn as nn
from transformers import Dinov2Model
import cv2
import numpy as np
import logging
import os
from torchvision.transforms import functional as F # Use functional for transforms on tensors
from torchvision.transforms import Compose, Resize, Normalize, ToTensor # Import specific transforms
from typing import List, Tuple # Added Tuple

# Set up logging - basic config for standalone, main app may override
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deepfake_detector') # Specific logger name

class OptimizedDeepfakeDetector(nn.Module):
    def __init__(self, model_path="/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt"):
        """Initialize the fine-tuned DINOv2 model for deepfake detection on CPU."""
        super().__init__()
        # Explicitly set to CPU. Change to "cuda" if GPU is available and desired.
        self.device = torch.device("cpu")
        logger.info(f"Initializing DINOv2 Deepfake Detector on {self.device}...")
        try:
            # Load base DINOv2 model
            self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-small")

            # Define the classifier head (use config for input size)
            # Ensure classifier_input_size matches the output dimension of the DINOv2 model used
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

            # Load the state dictionary, ensuring it maps to the correct device (CPU in this case)
            state_dict = torch.load(model_path, map_location=self.device)
            self.load_state_dict(state_dict)

            # Move the entire model (DINOv2 + classifier) to the specified device
            self.to(self.device)
            self.eval() # Set to evaluation mode (crucial for consistent results and disabling dropout)

            # Define preprocessing (consistent for prediction and explanation)
            self.image_size = 224 # Standard for many ViTs, DINOv2 often uses this or larger
            # Normalization values common for models trained on ImageNet
            self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ToTensor() will be applied first to convert NumPy/PIL to Tensor C, H, W and scale [0,1]

            logger.info(f"DINOv2 Deepfake Detector initialized successfully from {model_path}")

        except FileNotFoundError: # Re-raise specific error for clarity
             logger.critical(f"Model weights file missing: {model_path}")
             raise
        except Exception as e:
            logger.critical(f"FATAL: Failed to initialize deepfake detector: {str(e)}", exc_info=True)
            # Depending on application context, you might exit or raise a more specific error
            raise RuntimeError(f"Deepfake detector initialization failed: {e}") from e

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor | None:
        """Internal helper to preprocess a single frame."""
        if frame is None or frame.size == 0:
            logger.warning("Attempted to preprocess an empty frame.")
            return None
        try:
            # 1. Convert BGR (OpenCV default) to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 2. Convert NumPy HWC to Tensor CHW and scale pixels to [0, 1]
            img_tensor = ToTensor()(img_rgb)
             # 3. Resize (expects CHW Tensor) using functional API for tensors
            # Use antialias=True for better quality resizing
            img_resized = F.resize(img_tensor, [self.image_size, self.image_size], antialias=True)
            # 4. Normalize (expects CHW Tensor)
            img_processed = self.normalize(img_resized)
            return img_processed
        except Exception as e:
            logger.error(f"Error during frame preprocessing: {e}", exc_info=True)
            return None

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]] | torch.Tensor:
        """
        Define the forward pass for the model.

        Args:
            x (torch.Tensor): Input batch of preprocessed image tensors (B, C, H, W).
            output_attentions (bool): If True, return attention weights along with logits.

        Returns:
            torch.Tensor: Logits (B, num_classes) if output_attentions is False.
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]: Logits and a tuple of attention tensors
                                                        (one per layer) if output_attentions is True.
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Pass the flag to the underlying DINOv2 model
        # DINOv2 returns a BaseModelOutputWithPooling or similar containing last_hidden_state, pooler_output, attentions etc.
        model_output = self.dinov2(x, output_attentions=output_attentions)

        # Use CLS token pooling (common ViT approach)
        # The CLS token's embedding is usually the first token in the sequence output
        # Shape: (batch_size, sequence_length, hidden_size) -> take [:, 0]
        features = model_output.last_hidden_state[:, 0]

        # Pass features through the classifier head
        logits = self.classifier(features)

        if output_attentions:
            # Return logits AND the tuple of attention tensors stored in the model output object
            # Make sure to access the correct attribute (might be .attentions or similar)
            attentions = model_output.attentions if hasattr(model_output, 'attentions') else None
            if attentions is None:
                 logger.warning("Requested attentions, but model output did not contain them.")
            return logits, attentions
        else:
            # Return only logits for standard prediction
            return logits

    @torch.no_grad() # Disable gradient calculations for inference
    def process_batch(self, image_tensors: List[torch.Tensor]) -> List[float]:
        """
        Processes a batch of already preprocessed image tensors.
        Assumes tensors are correctly sized, normalized, and on the correct device.

        Args:
            image_tensors (List[torch.Tensor]): List of preprocessed image tensors (C, H, W).

        Returns:
            List[float]: List of deepfake scores (probability of being fake - class 1).
                         Returns list of 0.0s with same length as input on error.
        """
        if not image_tensors:
            return []
        self.eval() # Ensure model is in evaluation mode

        try:
            # Stack tensors into a batch (B, C, H, W) and move to device
            batch_tensor = torch.stack(image_tensors).to(self.device)

            # Get logits from the forward pass (no attentions needed here)
            outputs = self(batch_tensor) # Calls forward(batch_tensor, output_attentions=False)

            # Apply softmax to get probabilities and select the score for class 1 (Fake)
            scores = torch.softmax(outputs, dim=1)[:, 1]

            # Move scores to CPU (if they were on GPU) and convert to numpy list
            return scores.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error processing preprocessed batch: {str(e)}", exc_info=True)
            # Return default score (0.0) for each input tensor on error
            return [0.0] * len(image_tensors)

    @torch.no_grad() # Disable gradient calculations for inference
    def predict_batch(self, frames: List[np.ndarray]) -> List[float]:
        """
        Predict deepfake scores for a batch of raw frames (BGR NumPy arrays).
        Handles ALL necessary preprocessing.

        Args:
            frames (List[np.ndarray]): List of input frames in BGR format.

        Returns:
            List[float]: List of deepfake scores, one per input frame.
                         Frames that fail preprocessing will have a score of 0.0.
        """
        if not frames:
            return []
        self.eval() # Ensure model is in evaluation mode

        preprocessed_tensors = []
        original_indices = [] # Keep track of which original frames were successfully processed

        for i, frame in enumerate(frames):
            processed_tensor = self._preprocess_frame(frame)
            if processed_tensor is not None:
                preprocessed_tensors.append(processed_tensor)
                original_indices.append(i) # Store index of successfully processed frame
            else:
                logger.warning(f"Preprocessing failed for frame at index {i}, skipping prediction for it.")

        # If no frames could be preprocessed, return all zeros
        if not preprocessed_tensors:
             logger.warning("No frames were successfully preprocessed in the batch.")
             return [0.0] * len(frames)

        # Process only the successfully preprocessed tensors using the internal batch processing method
        scores_processed = self.process_batch(preprocessed_tensors)

        # Map the calculated scores back to the original frame indices.
        # Initialize final scores list with 0.0 for all input frames.
        final_scores = [0.0] * len(frames)

        # Check if the number of scores matches the number of successfully processed frames
        if len(scores_processed) == len(original_indices):
            for i, score in enumerate(scores_processed):
                 orig_idx = original_indices[i] # Get the original index this score corresponds to
                 final_scores[orig_idx] = score
        else:
            # This indicates an unexpected internal error in process_batch or logic here.
            logger.error(f"Score mapping mismatch: Processed scores count ({len(scores_processed)}) "
                         f"!= successfully preprocessed indices count ({len(original_indices)}). "
                         f"Returning default scores (0.0) for all frames.")
            # Keep final_scores as all zeros in this error case.

        return final_scores

    # --- Explanation Methods ---

    @torch.enable_grad() # Enable gradients specifically for explanation methods if needed (e.g., GradCAM)
                       # For attention maps from forward pass, gradients might not be strictly necessary,
                       # but keeping it doesn't hurt unless memory is extremely tight.
                       # Changed to torch.no_grad() as we only need forward pass attentions.
    @torch.no_grad()
    def get_attention_maps(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Generates attention map heatmap for a single frame (BGR NumPy array).
        Focuses on CLS token attention to patch tokens in the last layer, averaged over heads.

        Args:
            frame (np.ndarray): Input frame (BGR format).

        Returns:
            np.ndarray | None: A normalized [0, 1] heatmap (H, W matching model input size)
                               or None if an error occurs.
        """
        if frame is None or frame.size == 0:
             logger.error("Cannot get attention maps for empty frame.")
             return None
        self.eval() # Ensure evaluation mode
        logger.debug("Generating attention map...")

        try:
            # --- Preprocess single frame ---
            img_processed = self._preprocess_frame(frame)
            if img_processed is None:
                 logger.error("Preprocessing failed for attention map generation.")
                 return None

            # Add batch dimension and move to the correct device
            input_batch = img_processed.unsqueeze(0).to(self.device)

            # --- Get Model Output with Attentions ---
            # Call forward pass requesting attentions
            logits, attentions = self(input_batch, output_attentions=True)

            if attentions is None or not isinstance(attentions, tuple) or len(attentions) == 0:
                logger.error("Failed to retrieve attention weights from the model.")
                return None

            # --- Process Attention (Last Layer, Avg Heads, CLS to Patches) ---
            # attentions is a tuple, one element per layer. Get the last layer's attentions.
            last_layer_attention = attentions[-1] # Shape: (batch, num_heads, seq_len, seq_len)
                                                # Here batch=1

            # Verify shape
            if last_layer_attention.dim() != 4 or last_layer_attention.shape[0] != 1:
                 logger.error(f"Unexpected attention tensor shape: {last_layer_attention.shape}")
                 return None

            num_heads = last_layer_attention.shape[1]
            seq_len = last_layer_attention.shape[2] # Should be 1 (CLS token) + num_patches
            num_patches = seq_len - 1

            # Average attention scores across all heads
            avg_attention = torch.mean(last_layer_attention, dim=1).squeeze(0) # Shape: (seq_len, seq_len)

            # Extract attention weights from the CLS token (index 0) to all patch tokens (indices 1 to end)
            cls_to_patch_attention = avg_attention[0, 1:] # Shape: (num_patches,)

            # --- Reshape Attention to 2D Grid ---
            # Calculate the grid size (assuming square image and square patches)
            # patch_size is needed here. DINOv2 often uses 14 or 16. Get from config if possible.
            patch_size = getattr(self.dinov2.config, 'patch_size', 14) # Default to 14 if not in config
            grid_size = self.image_size // patch_size # e.g., 224 // 14 = 16

            # Verify that the number of patches matches the grid size calculation
            if grid_size * grid_size != num_patches:
                logger.warning(f"Calculated grid size ({grid_size}x{grid_size}) using patch size {patch_size} "
                               f"doesn't match number of patches from attention ({num_patches}). "
                               f"Attempting sqrt({num_patches}).")
                # Attempt to infer grid size from the number of patches if square
                grid_size_from_patches = int(np.sqrt(num_patches))
                if grid_size_from_patches * grid_size_from_patches == num_patches:
                     grid_size = grid_size_from_patches
                     logger.info(f"Using inferred grid size: {grid_size}x{grid_size}")
                else:
                     logger.error(f"Cannot determine a valid square grid size for {num_patches} patches.")
                     return None # Cannot reshape if grid size is wrong

            # Reshape the 1D attention vector into a 2D grid
            attention_map = cls_to_patch_attention.reshape(grid_size, grid_size)

            # Move map to CPU and convert to NumPy for OpenCV operations
            attention_map_np = attention_map.cpu().numpy()

            # --- Upscale and Normalize Heatmap ---
            # Upscale the small grid attention map to the model's input image size using bilinear interpolation
            heatmap_resized = cv2.resize(attention_map_np,
                                         (self.image_size, self.image_size),
                                         interpolation=cv2.INTER_LINEAR)

            # Normalize the heatmap values to the range [0, 1] for visualization
            min_val, max_val = np.min(heatmap_resized), np.max(heatmap_resized)
            if max_val > min_val:
                 heatmap_normalized = (heatmap_resized - min_val) / (max_val - min_val)
            else: # Handle cases where the heatmap is flat (all values the same)
                 heatmap_normalized = np.zeros_like(heatmap_resized)

            logger.debug("Attention map generated and normalized successfully.")
            return heatmap_normalized # Return the normalized heatmap (H, W) as a NumPy array

        except Exception as e:
            logger.error(f"Error generating attention map: {e}", exc_info=True)
            return None

    def create_overlay(self,
                       frame: np.ndarray,
                       heatmap: np.ndarray,
                       colormap: int = cv2.COLORMAP_JET,
                       alpha: float = 0.5) -> np.ndarray | None:
        """
        Overlays a normalized heatmap [0, 1] onto the original frame.

        Args:
            frame (np.ndarray): Original frame (BGR format).
            heatmap (np.ndarray): Normalized heatmap [0, 1] (H, W - should match model input size, e.g., 224x224).
            colormap (int): OpenCV colormap constant (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_HOT).
            alpha (float): Blending factor for the heatmap (0.0 = original frame, 1.0 = heatmap only).

        Returns:
            np.ndarray | None: BGR image with overlay, resized to match heatmap dimensions, or None on error.
        """
        if heatmap is None:
            logger.warning("Cannot create overlay with None heatmap.")
            return None
        if frame is None or frame.size == 0:
            logger.error("Cannot create overlay with empty frame.")
            return None
        if not (0.0 <= alpha <= 1.0):
             logger.warning(f"Alpha value {alpha} out of range [0, 1]. Clamping to 0.5.")
             alpha = 0.5
        # Ensure heatmap is 2D
        if heatmap.ndim != 2:
             logger.error(f"Heatmap must be 2D, but got shape {heatmap.shape}")
             return None

        try:
            # Resize original frame to match the heatmap's dimensions for accurate blending
            # Heatmap dimensions should match self.image_size (e.g., 224x224)
            target_h, target_w = heatmap.shape
            frame_resized = cv2.resize(frame, (target_w, target_h))

            # Convert the normalized [0, 1] heatmap to an 8-bit grayscale image [0, 255]
            heatmap_uint8 = np.uint8(255 * heatmap)

            # Apply the specified colormap to the 8-bit heatmap
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

            # Blend the resized original frame and the colored heatmap
            # overlay = alpha * colored_heatmap + (1 - alpha) * frame_resized
            overlay = cv2.addWeighted(frame_resized, 1 - alpha, colored_heatmap, alpha, 0)

            return overlay

        except Exception as e:
             logger.error(f"Error creating overlay: {e}", exc_info=True)
             return None

# --- Standalone Test Section ---
if __name__ == "__main__":
    print("--- Running Standalone Deepfake Detector Test ---") # Use print for standalone

    # --- Configuration (VERIFY PATHS!) ---
    MODEL_WEIGHTS_PATH = "/Users/kartik/Desktop/vs/VeriStream/backend/dinov2_deepfake_final.pt"
    TEST_IMAGE_PATH = "/Users/kartik/Desktop/vs/VeriStream/backend/test2.png" # Ensure this image exists
    OUTPUT_OVERLAY_FILENAME = "attention_overlay_output.jpg" # Output filename

    # --- Initialization ---
    print(f"Initializing detector with weights: {MODEL_WEIGHTS_PATH}")
    try:
        detector = OptimizedDeepfakeDetector(model_path=MODEL_WEIGHTS_PATH)
        print(f"Detector initialized successfully on device: {detector.device}")
    except Exception as init_err:
        print(f"\nFATAL ERROR: Failed to initialize detector.")
        print(f"Reason: {init_err}")
        if isinstance(init_err, FileNotFoundError):
             print("Please ensure the model weights file exists at the specified path.")
        exit(1) # Exit if initialization fails

    # --- Load Test Image ---
    print(f"Loading test image: {TEST_IMAGE_PATH}")
    if not os.path.exists(TEST_IMAGE_PATH):
         print(f"\nERROR: Test image not found at '{TEST_IMAGE_PATH}'!")
         exit(1)

    test_frame = cv2.imread(TEST_IMAGE_PATH)
    if test_frame is None:
        print(f"\nERROR: Failed to load test image using OpenCV. Check file path and format.")
        exit(1)
    else:
         print(f"Test image loaded (Shape: {test_frame.shape})")

    # --- 1. Test Prediction Score ---
    print("\n--- Testing Prediction ---")
    try:
        scores = detector.predict_batch([test_frame]) # Pass as a list
        if scores and len(scores) == 1:
            score = scores[0]
            print(f"Predicted Deepfake Score: {score:.4f}")
            verdict = "FAKE" if score > 0.5 else "REAL"
            print(f"Classification (@0.5 threshold): {verdict}")
        else:
            print(f"ERROR: Prediction returned unexpected result: {scores}")
    except Exception as pred_err:
        print(f"ERROR during prediction test: {pred_err}")
        # Optionally print traceback for debugging
        # import traceback; traceback.print_exc()

    # --- 2. Test Explanation Generation (Attention Map) ---
    print("\n--- Testing Explanation (Attention Map Generation) ---")
    heatmap = None # Initialize heatmap to None
    try:
        heatmap = detector.get_attention_maps(test_frame)
        if heatmap is not None:
            print(f"Heatmap generated successfully (Shape: {heatmap.shape}, "
                  f"Range: [{heatmap.min():.2f}, {heatmap.max():.2f}])")
        else:
            print("ERROR: Heatmap generation failed (returned None). Check logs for details.")
    except Exception as explain_err:
        print(f"ERROR during attention map generation test: {explain_err}")
        # import traceback; traceback.print_exc()

    # --- 3. Test Overlay Creation ---
    if heatmap is not None: # Only proceed if heatmap generation was successful
        print("\n--- Testing Explanation (Overlay Creation) ---")
        overlay_image = None # Initialize overlay_image to None
        try:
            # You can change alpha (0.0-1.0) and colormap (cv2.COLORMAP_...)
            overlay_image = detector.create_overlay(test_frame, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET)
            if overlay_image is not None:
                print(f"Overlay image created successfully (Shape: {overlay_image.shape})")

                # --- 4. Save Output ---
                print(f"Attempting to save overlay image to: {OUTPUT_OVERLAY_FILENAME}")
                try:
                    # Use imwrite to save the BGR overlay image
                    success = cv2.imwrite(OUTPUT_OVERLAY_FILENAME, overlay_image)
                    if success:
                        print(f"Successfully saved attention overlay to: {os.path.abspath(OUTPUT_OVERLAY_FILENAME)}")
                    else:
                        # This can happen due to permissions, invalid path components, etc.
                        print(f"ERROR: Failed to save overlay image (cv2.imwrite returned False).")
                except Exception as save_err:
                    # Catch potential exceptions during file writing (e.g., disk full)
                    print(f"ERROR: Exception occurred while saving overlay image to '{OUTPUT_OVERLAY_FILENAME}': {save_err}")
            else:
                print("ERROR: Overlay creation failed (returned None). Check logs for details.")
        except Exception as overlay_err:
            print(f"ERROR during overlay creation test: {overlay_err}")
            # import traceback; traceback.print_exc()
    else:
        print("\nSkipping Overlay Creation because heatmap generation failed.")

    print("\n--- Standalone Test Finished ---")