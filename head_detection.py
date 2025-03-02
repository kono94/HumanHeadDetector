"""Object Detection Module using ONNX model.

This module provides a class `HeadDetector` that performs object detection on images
using a pre-trained ONNX model with a fixed path.

Usage:
    from head_detector import HeadDetector

    detector = HeadDetector()
    labels, boxes, scores = detector.detect('path/to/image.jpg', output_image_path='output.jpg')
"""

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

class HeadDetector:
    """A class for performing object detection using an ONNX model."""

    # Fixed path to the ONNX model (adjust as needed)
    DEFAULT_MODEL_PATH = 'head_model_640.onnx'

    def __init__(self, onnx_model_path=DEFAULT_MODEL_PATH, confidence_threshold=0.2):
        """
        Initialize the HeadDetector with a model path and detection threshold.

        Args:
            onnx_model_path (str): Path to the ONNX model file. Defaults to 'model.onnx'.
            confidence_threshold (float): Detection confidence threshold. Defaults to 0.2.
        """
        self.sess = ort.InferenceSession(onnx_model_path)
        self.confidence_threshold = confidence_threshold
        self.target_size = (640, 640)  # Define target size for resizing

    def _preprocess_image(self, im_pil):
        """
        Preprocess the image by resizing and converting to a NumPy array.

        Args:
            im_pil (PIL.Image): The input PIL Image.

        Returns:
            np.ndarray: Preprocessed image array with shape (1, C, H, W).
        """
        # Resize the image to target size
        im_resized = im_pil.resize(self.target_size, Image.LANCZOS)

        # Convert PIL Image to NumPy array (H, W, C)
        im_array = np.array(im_resized, dtype=np.float32)

        # Transpose to (1, C, H, W)
        im_array = np.expand_dims(np.transpose(im_array / 255.0, (2, 0, 1)), axis=0)

        return im_array

    def detect(self, image, confidence_threshold=None, output_image_path=None):
        """
        Perform object detection on the specified image.

        Args:
            image (str or PIL.Image.Image): Path to the input image file or a PIL Image object.
            confidence_threshold (float, optional): Detection confidence threshold.
                                                Takes self.confidence_threshold if not given.
            output_image_path (str, optional): Path to save the output image with blurred regions.
                                            If None, no image is saved.

        Returns:
            tuple: (labels, boxes, scores)
                - labels: NumPy array of detected class labels.
                - boxes: NumPy array of bounding box coordinates.
                - scores: NumPy array of confidence scores.

        Raises:
            TypeError: If image is neither a string nor a PIL Image object.
        """
        # Handle the input based on its type
        if isinstance(image, str):
            im_pil = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            im_pil = image.convert('RGB')
        else:
            raise TypeError("image must be a string (path to image) or a PIL Image object")

        w, h = im_pil.size
        orig_size = np.array([[w, h]], dtype=np.int64)  # Shape (1, 2), dtype=int64 as expected

        im_data = self._preprocess_image(im_pil)

        # Run inference
        output = self.sess.run(
            output_names=None,
            input_feed={'images': im_data, "orig_target_sizes": orig_size}
        )

        # Unpack detection results (output should be NumPy arrays from onnxruntime)
        _, boxes, scores = output

        # Filter detections based on threshold
        scr = scores[0]  # Single image, so take the first batch
        mask = scr > (confidence_threshold if confidence_threshold is not None else self.confidence_threshold)
        box = boxes[0][mask]
        scr = scr[mask]

        if output_image_path:
            processed_image = self.blur_heads(im_pil, box)
            processed_image.save(output_image_path)

        return box, scr

    def draw_image(self, image, labels, boxes):
        """
        Draw bounding boxes and labels on the image.

        Args:
            image (PIL.Image): The input image to draw on.
            labels (np.ndarray): Detected class labels.
            boxes (np.ndarray): Bounding box coordinates.

        Returns:
            PIL.Image: The image with bounding boxes and labels drawn.
        """
        draw = ImageDraw.Draw(image)
        for b, l in zip(boxes, labels):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=str(int(l)), fill='blue')  # Convert label to int
        return image

    def blur_heads(self, image, boxes, padding_factor=0.1, blur_factor=15):
        """
        Blur the head regions in the image with added padding around each bounding box.

        Args:
            image (PIL.Image): The input image.
            boxes (np.ndarray): Bounding box coordinates as [x1, y1, x2, y2].
            padding_factor (float): Factor to expand the bounding box (e.g., 0.1 = 10%).
            blur_factor (float): Radius for Gaussian blur (default 15).

        Returns:
            PIL.Image: The image with blurred head regions including padding.
        """
        for box in boxes:
            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            pad_width = padding_factor * width
            pad_height = padding_factor * height

            x1_new = max(0, x1 - pad_width)
            y1_new = max(0, y1 - pad_height)
            x2_new = min(image.width, x2 + pad_width)
            y2_new = min(image.height, y2 + pad_height)

            left = int(x1_new)
            upper = int(y1_new)
            right = int(x2_new)
            lower = int(y2_new)

            # Skip invalid regions
            if left >= right or upper >= lower:
                continue

            # Crop the expanded region
            region = image.crop((left, upper, right, lower))

            # Apply Gaussian blur
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_factor))

            # Paste the blurred region back
            image.paste(blurred_region, (left, upper))

        return image

if __name__ == '__main__':
    detector = HeadDetector()
    image = Image.open("examples/people3.jpg").convert('RGB')
    boxes, scores = detector.detect(image, confidence_threshold=0.2)
    blurred_image = detector.blur_heads(image, boxes, padding_factor=0.1, blur_factor=20)
    blurred_image.save("output_blurred.jpg")