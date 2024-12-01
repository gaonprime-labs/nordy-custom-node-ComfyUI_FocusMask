# focus_mask.py
import cv2
import numpy as np
import torch

class FocusMaskExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image input
                "method": (["laplacian", "sobel", "modified_laplacian"],),
                "blur_size": ("INT", {"default": 9, "min": 3, "max": 21, "step": 2}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "extract_focus_mask"
    CATEGORY = "image/processing"
    OUTPUT_NODE = False

    def extract_focus_mask(self, image, method="laplacian", blur_size=9):
        # Convert from ComfyUI image format (torch tensor) to numpy
        if isinstance(image, torch.Tensor):
            # Ensure image is on CPU and convert to numpy
            image = image.cpu().numpy()
            # Convert from BCHW to HWC format and scale to 0-255
            image = (image[0].transpose(1, 2, 0) * 255).astype(np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image

        if method == "laplacian":
            # Laplacian variance method
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            focus_map = np.abs(laplacian)

        elif method == "sobel":
            # Sobel method
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            focus_map = np.sqrt(sobelx**2 + sobely**2)

        else:  # modified_laplacian
            # Modified Laplacian method
            kernel = np.array([[-1, 2, -1]])
            ml_x = cv2.filter2D(gray.astype(float), -1, kernel)
            ml_y = cv2.filter2D(gray.astype(float), -1, kernel.T)
            focus_map = np.absolute(ml_x) + np.absolute(ml_y)

        # Apply Gaussian blur for smoothing
        if blur_size > 0:
            focus_map = cv2.GaussianBlur(focus_map, (blur_size, blur_size), 0)

        # Normalize to 0-1 range
        focus_map = (focus_map - focus_map.min()) / (focus_map.max() - focus_map.min() + 1e-8)

        # Convert back to torch tensor format for ComfyUI
        focus_map = torch.from_numpy(focus_map).float()
        # Add batch and channel dimensions
        focus_map = focus_map.unsqueeze(0).unsqueeze(0)

        return (focus_map,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FocusMaskExtractor": FocusMaskExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FocusMaskExtractor": "Extract Focus Mask"
}
