import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

data_dir = "data"
classes = ["benign", "malignant"]

# Count & stats
sizes = {cls: [] for cls in classes}
for cls in classes:
    path = os.path.join(data_dir, cls)
    count = len(os.listdir(path))
    print(f"{cls}: {count} images")
    for img_name in os.listdir(path):
        img = Image.open(os.path.join(path, img_name))
        sizes[cls].append(img.size)

# Avg size
for cls in classes:
    if sizes[cls]:
        avg_w, avg_h = np.mean([s[0] for s in sizes[cls]]), np.mean([s[1] for s in sizes[cls]])
        print(f"{cls} avg size: {avg_w:.1f}x{avg_h:.1f}")

# Display samples (up to 5 per class)
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for i, cls in enumerate(classes):
    imgs = os.listdir(os.path.join(data_dir, cls))[:5]
    for j, img_name in enumerate(imgs):
        img_path = os.path.join(data_dir, cls, img_name)
        image = Image.open(img_path)
        axs[i, j].imshow(image)
        axs[i, j].axis("off")
        axs[i, j].set_title(f"{cls} ({image.size})")
    # Hide empty subplots
    for j in range(len(imgs), 5):
        axs[i, j].axis("off")

plt.tight_layout()
plt.savefig("eda_samples.png")  # Save for easy view
plt.show()