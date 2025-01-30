import imageio
import os
import re

# Directory containing the images
image_dir = "figs/bending_test_litho"

# Output GIF file name
output_gif = "figs/bending_test_litho/output.gif"

# Collect and sort image files by ID
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith("_gif.jpg")],
    key=lambda x: int(re.search(r"bending_test_litho_(\d+)_fwd_trans_gif", x).group(1))
)
image_files = image_files[:20]

# Read images and write to GIF
with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = imageio.imread(img_path)
        writer.append_data(image)

print(f"GIF saved as {output_gif}")