# Focus Detection Nodes for ComfyUI

This custom node pack adds two specialized focus detection nodes to ComfyUI:

## Nodes

### Focus Outline
Generates an edge-based visualization of focus changes in an image. This node highlights the boundaries between in-focus and out-of-focus areas, creating a line-art-like representation of focus transitions.

Parameters:
- **Method**: Choose between different focus detection algorithms
  - Laplacian: Best for general use, detects sharp changes in focus
  - Sobel: Alternative edge detection, can be more robust to noise
  - Modified Laplacian: Enhanced detection for certain types of images
- **Blur Size**: Controls the smoothness of the detected edges (3-21, odd numbers only)

### Focus Mask
Creates a binary mask separating in-focus areas (white) from out-of-focus areas (black). Useful for isolating sharp regions or creating masks for further processing.

Parameters:
- **Threshold**: Controls the cutoff between what's considered in/out of focus (0-1)
  - Lower values (0.05-0.1) typically give best results
  - Adjust based on image content and desired sensitivity
- **Blur Size**: Smooths the mask edges (3-21, odd numbers only)
- **Denoise**: Reduces image noise before focus detection (0-10)
  - Higher values = more aggressive noise reduction
  - 0 = no denoising
  - Useful for noisy images or when getting speckled results
- **Sensitivity**: Controls how aggressively focus is detected (0.1-5.0)
  - Higher values = more areas considered in focus
  - Lower values = stricter focus detection

## Installation

1. Create a `custom_nodes` folder in your ComfyUI installation if it doesn't exist
2. Clone this repository into the custom_nodes folder by opening the terminal, and using these inputs:
- cd custom_nodes
- git clone https://github.com/risunobushi/ComfyUI_FocusMask.git
3. Install requirements:
open the terminal in your custom_nodes/ComfyUI_FocusMask folder, and input:
- pip install -r requirements.txt
4. Restart ComfyUI

## Requirements
- OpenCV (opencv-python)
- NumPy
- PyTorch (typically already installed with ComfyUI)

## Use Cases
Focus Outline:
Visualizing focus fall-off in images
Detecting focus transitions
Creating artistic edge effects based on focus
Analyzing lens characteristics

Focus Mask:
Isolating in-focus subjects
Creating masks for selective editing
Depth estimation assistance
Focus stacking preparation
Quality control for image sharpness
