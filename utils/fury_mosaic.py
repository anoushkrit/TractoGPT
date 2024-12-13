#%%
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from fury import window, actor
import pickle
from dipy.tracking.distances import approx_polygon_track
from nibabel.streamlines.array_sequence import ArraySequence
def load_tractogram(trk_file):
    """Load tractogram streamlines from a .trk file."""
    tractogram = nib.streamlines.load(trk_file)
    return tractogram.streamlines

def save_tractogram_view(streamlines, output_path, zoom_level=20):
    """Save a zoomed-in, axial view of a tractogram with a black background as a PNG."""
    scene = window.Scene()
    scene.background((0, 0, 0))
    streamline_actor = actor.line(streamlines, linewidth=0.2)
    scene.add(streamline_actor)
    scene.reset_camera()
    camera = scene.camera()
    camera.parallel_scale = zoom_level
    scene.set_camera(position=(0, 500, 0), focal_point=(0, 0, 0), view_up=(0, 0, 1))
    scene.zoom(1.6)
    window.snapshot(scene, fname=output_path, size=(500, 500), offscreen=True)
    scene.clear()


def crop_black_borders(img, threshold=10, min_size=(1, 1)):
    """Crop out the black borders of an image, retaining only the region containing color.
    
    Parameters:
        img (numpy.ndarray): The input image array.
        threshold (int): The threshold to determine what is considered 'black'.
        min_size (tuple): Minimum width and height for the cropped area.
        
    Returns:
        numpy.ndarray: The cropped image.
    """
    # Find the bounding box of non-black pixels
    mask = np.any(img > threshold, axis=-1)  # Create a mask for non-black pixels
    if not np.any(mask):  # If the image is completely black
        return img
    
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # Add 1 to include the border pixels

    # Calculate width and height of the cropped area
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # Check if the cropped area meets the minimum size requirements
    if crop_width < min_size[1] or crop_height < min_size[0]:
        return img  # Return original image if cropped area is too small

    return img[y_min:y_max, x_min:x_max]

def create_mosaic_image(tractograms, rows, cols, output_file="mosaic.png"):
    """Create a mosaic of tractogram images on a black background with customized row and column labels."""
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10), facecolor='black')
    common_base = os.path.commonprefix([os.path.basename(f) for f in tractograms])
    tractograms.sort(key=lambda x: os.path.basename(x))
    
    # Define row labels
    row_labels = ["FIESTA", "Ground Truth", "TractoGPT-hcp", "TractoGPT"]

    for i, (trk_file, ax) in enumerate(zip(tractograms, axes.flat)):
        img_path = f"temp_view_{i}.png"

        if trk_file.endswith(".trk"):
            streamlines = load_tractogram(trk_file)
        else:
            with open(trk_file, 'rb') as f:
                streamlines = pickle.load(f)
            streamlines = list(map(lambda x: approx_polygon_track(ArraySequence(x)), streamlines))

        save_tractogram_view(streamlines, img_path, zoom_level=100)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.axis("off")
    
    # Add column labels
    for i, ax in enumerate(axes[0]):
        ax.set_title(os.path.basename(tractograms[i]).split("_")[1].replace('.trk', ''), color='white', fontsize=14)
    
    # Add row labels with tilt and font size
    for j, ax in enumerate(axes[:, 0]):
        ax.set_ylabel(row_labels[j], rotation=90, color='white', labelpad=45, va='center', fontsize=14)

    # Remove any remaining empty subplots
    for j in range(i + 1, rows * cols):
        axes.flat[j].axis("off")

    # Adjust layout for minimal spacing
    plt.tight_layout()  # Automatically adjust layout
    plt.subplots_adjust(wspace=0, hspace=0)  # No space between images
    plt.savefig(output_file, dpi=300, facecolor='black', bbox_inches="tight", pad_inches=0)
    plt.show()



tractograms = [
    os.path.join('./', f) 
    for f in os.listdir('./') 
    if f.endswith(('.trk', '.pkl'))
]
rows = 4
cols = 4

create_mosaic_image(tractograms, rows, cols)
