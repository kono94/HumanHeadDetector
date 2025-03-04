# HeadDetector: Simple Head Detection with ONNX

**HumanHeadDetector** is a lightweight Python module for detecting heads in images using a pre-trained ONNX model. It provides a simple `HeadDetector` class that can locate heads in an image and optionally blur the detected regions for privacy purposes. The module is designed to be easy to use, requiring minimal setup and only using `onnxruntime` and `Pillow` as main dependencies. It is based on **RT-DETRv2**.

## Features
- Detect heads in images using a pre-trained ONNX model.
- Accepts both image file paths and PIL Image objects as input.
- Option to blur detected head regions with customizable padding and blur intensity.
- Simple API for integration into larger projects.
- Lightweight with minimal dependencies.

## Examples
The following examples show the original images alongside their blurred versions after head detection with `HeadDetector`.
The following examples show the original images alongside their blurred versions after head detection with `HeadDetector`.

| Original                                  | Blurred                                   |
|-------------------------------------------|-------------------------------------------|
| ![Original Image 1](examples/people.jpg?raw=true&width=300) | ![Blurred Image 1](examples/people_blurred.jpg?raw=true&width=300) |
| ![Original Image 2](examples/people2.jpg?raw=true&width=300) | ![Blurred Image 2](examples/people2_blurred.jpg?raw=true&width=300) |
| ![Original Image 3](examples/people3.jpg?raw=true&width=300) | ![Blurred Image 3](examples/people3_blurred.jpg?raw=true&width=300) |
| ![Original Image 4](examples/people4.jpg?raw=true&width=300) | ![Blurred Image 4](examples/people4_blurred.jpg?raw=true&width=300) |

## Requirements
This project requires Python 3.6+ and the following dependencies (listed in `requirements.txt`):
- `onnxruntime` (for ONNX model inference)
- `Pillow` (PIL fork, for image handling and processing)

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/head-detector.git
cd head-detector
```

### 2. Set Up a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies. Follow these steps to create and activate one:

#### On Linux/MacOS
```bash
python3 -m venv env
source env/bin/activate
```

You’ll see (env) in your terminal, indicating the virtual environment is active.

### 3. Install Dependencies
Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage
The `HeadDetector` class provides a simple API for detecting heads in images and optionally blurring the detected regions. Here are some examples:

#### Basic Detection
Detect heads in an image and get the labels, bounding boxes, and scores:

```python
from head_detector import HeadDetector

# Initialize the detector
detector = HeadDetector()

# Detect heads in an image (provide path or PIL Image)
boxes, scores = detector.detect("path/to/image.jpg")
print("Boxes:", boxes)
print("Scores:", scores)
```

#### Detection with Output Image
Detect heads and save an image with blurred heads:
```python
from head_detector import HeadDetector
from PIL import Image

# Initialize the detector
detector = HeadDetector()

# Load an image
image = Image.open("people.jpg").convert('RGB')

# Detect heads
boxes, scores = detector.detect(image)

# Blur detected heads and save the result
blurred_image = detector.blur_heads(image, boxes, padding_factor=0.2, blur_factor=20)
blurred_image.save("output_blurred.jpg")
```


## License
This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Contact
For questions or issues, please open an issue on GitHub.