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
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Load SAM2 model
sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Extract frames from video using ffmpeg (ensure you have ffmpeg installed and available in your PATH)
os.system("ffmpeg -i '/content/drive/MyDrive/Personal Files/sam_test.mp4' -q:v 2 -start_number 0 'Video_Test/%05d.jpg'")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# Directory containing extracted frames
video_dir = "/content/Video_Test"

# Scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Take a look at the first video frame
frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"Frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

ann_frame_idx = 0  # The frame index we interact with
ann_obj_id = 1  # Give a unique id to each object we interact with (it can be any integer)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[200, 500]], dtype=np.float32)
# For labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

# Assuming `inference_state` is defined elsewhere in the notebook/script
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"Frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

Feel free to adjust any specific sections or add more details as needed for your project.
