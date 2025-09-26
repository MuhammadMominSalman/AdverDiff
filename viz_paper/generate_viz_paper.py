import matplotlib.pyplot as plt
from skimage.io import imread
import os

image_paths = [
        "./test_imgs/clear_gt.jpg", "./test_imgs/seg_overcast.jpg", "./test_imgs/edge_overcast.jpg",
    "./test_imgs/ip2p_overcast.jpg",
    "./test_imgs/resized_result1.jpg", "./test_imgs/day_snowy_seg_6_result1.jpg", "./test_imgs/snowy_5_result1.jpg",
    "./test_imgs/snowy_day_ip2p_result1.jpg",

        "./test_imgs/resized_0066b72f-974f6883.jpg", "./test_imgs/night_rainy_seg_0_0066b72f-974f6883.jpg", "./test_imgs/night_rainy_edge_2_0066b72f-974f6883.jpg",
    "./test_imgs/rainy_night_ip2p_0_0066b72f-974f6883.jpg",
]

images = [imread(img_path) for img_path in image_paths]
captions = ["Ground Truth", "Segmentation", "Edge", "Ip2p"]

# Create figure with 3 rows and 4 columns
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Plot images
for idx, ax in enumerate(axes.flat):
    img = images[idx]
    ax.imshow(img)
    ax.axis('off')
    
    # Add caption only for top row (row 0)
    if idx < 4:
        ax.set_title(captions[idx], fontsize=12)

# Add vertical lines after the first image in each row
line_x = 1 / 4  # since there are 4 images
fig.subplots_adjust(wspace=0)
fig.canvas.draw()
line = plt.Line2D([line_x, line_x], [0, 1], color="black", linewidth=2, transform=fig.transFigure)
fig.add_artist(line)
# for row in range(3):
#     # Coordinates in figure-relative space
#     left_col_index = row * 4
#     right_col_index = left_col_index + 1
    
#     # Compute x position between first and second column of each row
#     bbox_left = axes[row, 0].get_position()
#     bbox_right = axes[row, 1].get_position()
#     line_x = (bbox_left.x1 + bbox_right.x0) / 2

#     # Add vertical line in figure coordinates
#     fig.add_artist(plt.Line2D(
#         [line_x, line_x], [bbox_left.y0, bbox_left.y1],
#         transform=fig.transFigure, color='black', linewidth=2
#     ))

plt.tight_layout()
plt.show()