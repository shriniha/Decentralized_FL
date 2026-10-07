# utils/dynamic_data.py
import os
import pandas as pd
import random
from pathlib import Path
import logging

logger = logging.getLogger("P2P-FL")

def ingest_new_data(incoming_dir='incoming'):
    """Load new samples from CSV + images dir. Returns list of (img_path, label)."""
    csv_path = Path(incoming_dir) / 'new_samples.csv'
    img_dir = Path(incoming_dir) / 'new_images'
    if not csv_path.exists() or not img_dir.exists():
        return []
    
    try:
        df = pd.read_csv(csv_path)
        # Assume cols: 'image_filename,label' (label: 0=benign, 1=malignant)
        new_samples = []
        for _, row in df.iterrows():
            img_file = row['image_filename']
            full_path = img_dir / img_file
            if full_path.exists():
                new_samples.append((str(full_path), int(row['label'])))
        # Clear for next ingest
        csv_path.unlink()
        logger.info(f"[DYNAMIC] Ingested {len(new_samples)} new samples from {csv_path}")
        return new_samples
    except Exception as e:
        logger.error(f"[DYNAMIC] Ingest failed: {e}")
        return []

def partition_new_samples(new_samples, num_clients):
    """Round-robin partition (IID). Returns list of lists per client."""
    random.shuffle(new_samples)
    partitions = [[] for _ in range(num_clients)]
    for i, sample in enumerate(new_samples):
        partitions[i % num_clients].append(sample)
    return partitions