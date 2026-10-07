import os
import random
import shutil

random.seed(42)  # Reproducibility

data_dir = "data"
clients_dir = "clients_data"
clients = ["0", "1", "2"]  # Numeric for Flower sim
split_ratio = 0.8  # 80/20 train/test per client

# Create folders
for client in clients:
    os.makedirs(os.path.join(clients_dir, client, "train", "benign"), exist_ok=True)
    os.makedirs(os.path.join(clients_dir, client, "train", "malignant"), exist_ok=True)
    os.makedirs(os.path.join(clients_dir, client, "test", "benign"), exist_ok=True)
    os.makedirs(os.path.join(clients_dir, client, "test", "malignant"), exist_ok=True)

# Non-IID split: 50/30/20
for cls in ["benign", "malignant"]:
    images = os.listdir(os.path.join(data_dir, cls))
    if not images:
        print(f"Error: No images in {data_dir}/{cls}! Download first.")
        continue
    random.shuffle(images)
    
    n = len(images)
    splits = [int(0.5 * n), int(0.3 * n), n - int(0.5 * n) - int(0.3 * n)]
    start = 0
    for client, size in zip(clients, splits):
        client_images = images[start:start + size]
        train_end = int(len(client_images) * split_ratio)
        train_imgs, test_imgs = client_images[:train_end], client_images[train_end:]
        
        # Copy train
        for img_name in train_imgs:
            src = os.path.join(data_dir, cls, img_name)
            dst = os.path.join(clients_dir, client, "train", cls, img_name)
            shutil.copy(src, dst)
        
        # Copy test
        for img_name in test_imgs:
            src = os.path.join(data_dir, cls, img_name)
            dst = os.path.join(clients_dir, client, "test", cls, img_name)
            shutil.copy(src, dst)
        
        print(f"Client {client} {cls}: {len(train_imgs)} train, {len(test_imgs)} test")
        start += size

print("Partitioning done! Check clients_data/0/train/benign/ etc. for files.")