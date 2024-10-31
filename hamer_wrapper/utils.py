from typing import List, Tuple, Dict, Any, Union, Optional
import sys, os
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm
import torch
import argparse
import av

PROJ_ROOT = Path(__file__).parent.parent
MARK_COLORS = [
    (255, 0, 0),  # red, right hand
    (0, 255, 0),  # green, left hand
]

def add_path(path: str) -> None:
    path_to_add = Path(path)
    if path_to_add.exists() and str(path_to_add) not in sys.path:
        sys.path.insert(0, str(path_to_add))


def make_clean_folder(folder):
    import shutil

    if Path(folder).exists():
        shutil.rmtree(folder)
    Path(folder).mkdir(parents=True, exist_ok=True)


def read_data_from_json(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def draw_landmarks(img, landmarks, color=(0, 255, 0), radius=3):
    vis = img.copy()
    for landmark in landmarks:
        if np.all(landmark == -1):
            continue
        cv2.circle(vis, tuple(landmark), radius, color, -1)
    return vis


def read_cv_image(img_path, idx=None):
    img = cv2.imread(str(img_path))
    if idx is None:
        return img
    return img, idx


def read_rgb_image(image_path, idx=None):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if idx is not None:
        return img, idx
    return img


def write_cv_image(img_path, img):
    cv2.imwrite(str(img_path), img)


def write_rgb_image(image_path, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(image_path), rgb)


def create_video_from_rgb_images(
    file_path: Union[str, Path], rgb_images: List[np.ndarray], fps: int = 30
) -> None:
    """Create a video from a list of RGB images."""
    if not rgb_images:
        raise ValueError("The list of RGB images is empty.")
    height, width = rgb_images[0].shape[:2]
    container = None
    try:
        container = av.open(str(file_path), mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.thread_type = "FRAME"  # Parallel processing of frames
        stream.thread_count = os.cpu_count()  # Number of threads to use
        for image in rgb_images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception as e:
        raise IOError(f"Failed to write video to '{file_path}': {e}")
    finally:
        if container:
            container.close()


def _plot_image(ax, image, name, facecolor, titlecolor, fontsize):
    """Helper function to plot an image in the grid."""
    if image.ndim == 3 and image.shape[2] == 3:  # RGB image
        ax.imshow(image)
    elif image.ndim == 2 and image.dtype == np.uint8:  # Grayscale/mask image
        unique_values = np.unique(image)
        cmap = "tab10" if len(unique_values) <= 10 else "gray"
        ax.imshow(image, cmap=cmap)
    elif image.ndim == 2 and image.dtype == bool:  # Binary image
        ax.imshow(image, cmap="gray")
    else:  # Depth or other image
        ax.imshow(image, cmap="viridis")

    if name:
        ax.text(
            5,
            5,
            name,
            fontsize=fontsize,
            color=titlecolor,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor=facecolor, alpha=0.5, edgecolor="none", pad=3),
        )


def draw_image_grid(
    images: List[np.ndarray],
    names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (1920, 1080),
    max_cols: int = 4,
    facecolor: str = "white",
    titlecolor: str = "black",
    fontsize: int = 12,
    bar_width: int = 0.2,
) -> np.ndarray:
    """Display a list of images in a grid and draw the title name on each image's top-left corner."""
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided to display.")
    num_cols = min(num_images, max_cols)
    num_rows = (num_images + num_cols - 1) // num_cols
    # Default to no names if not provided
    if names is None or len(names) != num_images:
        names = [None] * num_images
    # Create figure and axis grid
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(figsize[0] / 100.0, figsize[1] / 100.0),
        dpi=100,
        facecolor=facecolor,
    )
    axs = np.atleast_1d(axs).flat  # Ensure axs is always iterable
    # Plot each image
    for i, (image, name) in enumerate(zip(images, names)):
        _plot_image(axs[i], image, name, facecolor, titlecolor, fontsize)
        axs[i].axis("off")
    # Hide unused axes
    for ax in axs[i + 1 :]:
        ax.axis("off")
    # Adjust layout and spacing
    plt.tight_layout(pad=bar_width, h_pad=bar_width, w_pad=bar_width)
    # Convert the figure to an RGB array
    fig.canvas.draw()
    rgb_image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    # Close the figure
    plt.close(fig)
    return rgb_image
