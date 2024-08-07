# SAM2-image-video-segmentation
Segment Anything 2 (SAM 2) is a real-time image and video segmentation model. It supports automatic mask generation to segment all objects and a point prompt feature for targeted segmentation. SAM 2 can identify specific objects in images and videos, making it ideal for precise object identification and segmentation tasks.

## Features

- **Automatic Mask Generator**: Segments all objects in an image or video and generates corresponding masks.
- **Point Prompt**: Allows for specific object segmentation using a point or series of points, or a bounding box.

## Requirements

To run the SAM 2 implementation, you will need the following libraries:

- OpenCV
- Pandas
- NumPy
- PyTorch (for model inference)
- Matplotlib (for visualizing results)

You can install the required libraries using pip:

```bash
pip install opencv-python pandas numpy torch matplotlib
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/mushaid01/segment-anything-2.git
    cd segment-anything-2
    ```

2. Open the Jupyter notebook:
    ```bash
    jupyter notebook SAM2_IMPLE.ipynb
    ```

3. Follow the instructions in the notebook to load your images or videos and perform segmentation using SAM 2.

## Running SAM 2

### Automatic Mask Generator

To use the automatic mask generator, simply load an image or video, and the model will generate masks for all objects.

### Point Prompt

To use the point prompt feature, provide specific points or a bounding box around the object you want to segment. The model will generate a mask for the specified object.

## Example Code

Here is a simple example of using SAM 2 in the notebook:

```python
import cv2
import torch
from sam2 import SAM2  # Assuming SAM2 is implemented in sam2.py

# Load the model
model = SAM2()

# Load an image
image = cv2.imread('path_to_image.jpg')

# Generate masks using the automatic mask generator
masks = model.generate_masks(image)

# Display the masks
for mask in masks:
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

Feel free to adjust any specific sections or add more details as needed for your project.
