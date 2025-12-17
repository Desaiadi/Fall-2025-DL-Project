"""
Create Kaggle Submission with YOUR Trained SSL Model
====================================================
This loads YOUR trained checkpoint and uses KNN for evaluation.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse


class FeatureExtractor:
    """Load YOUR trained SSL model"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        print(f"Loading YOUR trained model from: {checkpoint_path}")
        
        # Initialize DINOv2-small architecture (no pretrained weights!)
        config = Dinov2Config.from_pretrained('facebook/dinov2-small')
        self.model = Dinov2Model(config)
        
        # Load YOUR trained weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # FREEZE the encoder (required for competition!)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model = self.model.to(device)
        self.device = device
        
        # Image processor
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        
        print("✓ Model loaded and FROZEN")
    
    def extract_batch_features(self, images):
        """Extract features from batch of PIL Images"""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use CLS token features
        cls_features = outputs.last_hidden_state[:, 0]
        return cls_features.cpu().numpy()


class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function"""
    if len(batch[0]) == 3:
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


def extract_features(feature_extractor, dataloader, split_name):
    """Extract features from dataloader"""
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name}"):
        if len(batch) == 3:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch
        
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dim {features.shape[1]}")
    return features, labels, all_filenames


def train_knn(train_features, train_labels, val_features, val_labels, k=5):
    """Train KNN classifier"""
    print(f"\nTraining KNN classifier (k={k})...")
    
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='distance',
        metric='cosine',
        n_jobs=-1
    )
    
    knn.fit(train_features, train_labels)
    
    train_acc = knn.score(train_features, train_labels)
    val_acc = knn.score(val_features, val_labels)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Val Accuracy: {val_acc*100:.2f}%")
    
    return knn


def create_submission(test_features, test_filenames, knn, output_path):
    """Create submission.csv"""
    print("\nGenerating predictions...")
    predictions = knn.predict(test_features)
    
    submission = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Submission saved: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission)}")
    print(f"\nFirst 10 rows:")
    print(submission.head(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to YOUR trained .pth file')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Load CSVs
    print("\nLoading dataset...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Create datasets
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist(),
        args.resolution
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist(),
        args.resolution
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        None,
        args.resolution
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Load YOUR model
    feature_extractor = FeatureExtractor(args.checkpoint, device)
    
    # Extract features
    train_features, train_labels, _ = extract_features(feature_extractor, train_loader, 'train')
    val_features, val_labels, _ = extract_features(feature_extractor, val_loader, 'val')
    test_features, _, test_filenames = extract_features(feature_extractor, test_loader, 'test')
    
    # Train KNN
    knn = train_knn(train_features, train_labels, val_features, val_labels, args.k)
    
    # Create submission
    create_submission(test_features, test_filenames, knn, args.output)
    
    print("\n✓ DONE! Upload submission.csv to Kaggle")


if __name__ == "__main__":
    main()