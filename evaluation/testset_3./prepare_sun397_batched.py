"""
Batched SUN397 Dataset Preparation
Processes in chunks of 500 images to balance speed and memory
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
from datasets import load_dataset
import gc


def create_kaggle_dataset_batched(cache_dir, output_dir, resolution=96, batch_size=500, seed=42):
    """
    Process in batches to balance speed and memory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    print("="*60)
    print("Loading SUN397 (using cached data)...")
    print("="*60)
    
    # Load from cache
    dataset = load_dataset("tanganke/sun397", split="train", cache_dir=cache_dir)
    
    label_names = dataset.features['label'].names
    total_size = len(dataset)
    
    print(f"Total: {total_size} images, {len(label_names)} classes")
    
    # Create split indices
    indices = np.random.permutation(total_size)
    n_train = int(total_size * 0.70)
    n_val = int(total_size * 0.15)
    
    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train+n_val],
        'test': indices[n_train+n_val:]
    }
    
    print(f"\nSplits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Process each split in batches
    def save_split_batched(indices, split_name, save_labels=True):
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True, parents=True)
        
        all_metadata = []
        
        print(f"\nProcessing {split_name} in batches of {batch_size}...")
        
        for batch_start in tqdm(range(0, len(indices), batch_size), desc=f"{split_name} batches"):
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # Process this batch
            for i, dataset_idx in enumerate(batch_indices):
                try:
                    new_idx = batch_start + i
                    
                    # Load image
                    item = dataset[int(dataset_idx)]
                    img = item['image'].convert('RGB')
                    label = item['label']
                    
                    # Resize
                    img = img.resize((resolution, resolution), Image.BILINEAR)
                    
                    # Save
                    if save_labels:
                        filename = f"{new_idx:06d}_class{label:03d}.jpg"
                    else:
                        filename = f"{new_idx:06d}.jpg"
                    
                    img.save(split_dir / filename, quality=85)
                    
                    all_metadata.append({
                        'filename': filename,
                        'class_id': label,
                        'class_name': label_names[label]
                    })
                    
                except Exception as e:
                    print(f"\nError at index {dataset_idx}: {e}")
                    continue
            
            # Clear memory after each batch
            gc.collect()
        
        # Save CSV
        df = pd.DataFrame(all_metadata)
        if save_labels:
            df.to_csv(output_dir / f'{split_name}_labels.csv', index=False)
        else:
            df[['filename']].to_csv(output_dir / f'{split_name}_images.csv', index=False)
        
        return df
    
    # Process all splits
    train_df = save_split_batched(splits['train'], 'train', save_labels=True)
    val_df = save_split_batched(splits['val'], 'val', save_labels=True)
    test_df = save_split_batched(splits['test'], 'test', save_labels=False)
    
    # Create sample submission
    pd.DataFrame({
        'id': test_df['filename'],
        'class_id': 0
    }).to_csv(output_dir / 'sample_submission.csv', index=False)
    
    # Class mapping
    pd.DataFrame({
        'class_id': range(len(label_names)),
        'class_name': label_names
    }).to_csv(output_dir / 'class_mapping.csv', index=False)
    
    print(f"\n{'='*60}")
    print("✓ Complete!")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='./raw_data')
    parser.add_argument('--output_dir', type=str, default='./data')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    create_kaggle_dataset_batched(
        args.cache_dir,
        args.output_dir,
        args.resolution,
        args.batch_size,
        args.seed
    )


if __name__ == "__main__":
    main()