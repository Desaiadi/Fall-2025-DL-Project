"""
Create Kaggle Submission with YOUR MoCo v3 Model
================================================

This script uses YOUR trained SSL model (MoCo v3) for feature extraction
and KNN classifier for predictions.

Usage:
    python create_submission_moco.py \
        --checkpoint /path/to/checkpoint_epoch_49.pth \
        --data_dir ./data \
        --output submission.csv \
        --k 5
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import argparse
from torchvision import transforms


# ============================================================================
#                          YOUR MOCO V3 MODEL
# ============================================================================

class YourFeatureExtractor:
    """
    Feature extractor using YOUR trained MoCo v3 ViT-Small backbone.
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize feature extractor from YOUR checkpoint.
        
        Args:
            checkpoint_path: Path to your MoCo v3 checkpoint
            device: 'cuda' or 'cpu'
        """
        print(f"Loading YOUR model from: {checkpoint_path}")
        self.device = device
        
        # Build ViT-Small backbone (same as training)
        self.model = timm.create_model(
            'vit_small_patch16_224',
            pretrained=False,  # YOUR model, not pretrained!
            num_classes=0,
            img_size=96,
            patch_size=16,
            global_pool='token'
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract encoder_q weights from MoCo model
        state_dict = checkpoint['model_state_dict']
        
        # Filter to get only encoder_q weights
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder_q.'):
                # Remove 'encoder_q.' prefix
                new_key = key.replace('encoder_q.', '')
                encoder_state_dict[new_key] = value
        
        # Load weights
        self.model.load_state_dict(encoder_state_dict)
        
        # FREEZE the model (required for competition!)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model = self.model.to(device)
        
        # Define preprocessing (match training)
        self.transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded and FROZEN!")
        print(f"  Feature dimension: {self.model.num_features}")
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of PIL Images.
        
        Args:
            images: List of PIL Images
        
        Returns:
            features: numpy array of shape (batch_size, feature_dim)
        """
        # Preprocess images
        image_tensors = torch.stack([self.transform(img) for img in images])
        image_tensors = image_tensors.to(self.device)
        
        # Extract features (model is frozen!)
        with torch.no_grad():
            features = self.model(image_tensors)
        
        return features.cpu().numpy()


# ============================================================================
#                          DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    """Simple dataset for loading images"""
    
    def __init__(self, image_dir, image_list, labels=None):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of labels (optional, for train/val)
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load image (don't resize here - done in transform)
        image = Image.open(img_path).convert('RGB')
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    """
    Extract features from a dataloader.
    
    Args:
        feature_extractor: YourFeatureExtractor instance
        dataloader: DataLoader
        split_name: Name of split (for progress bar)
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (or None for test)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:  # train/val
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:  # test
            images, filenames = batch
        
        # Extract features for batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# ============================================================================
#                          KNN CLASSIFIER
# ============================================================================

def train_knn_classifier(train_features, train_labels, val_features, val_labels, k_values=[1, 5, 10, 20]):
    """
    Train KNN classifier on features and find best k.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels (N_train,)
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels (N_val,)
        k_values: List of k values to try
    
    Returns:
        best_classifier: Best KNN classifier
        best_k: Best k value
    """
    print(f"\nTuning KNN classifier...")
    
    best_val_acc = 0
    best_classifier = None
    best_k = None
    
    results = []
    
    for k in k_values:
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',  # Weight by inverse distance
            metric='cosine',  # Cosine similarity for embeddings
            n_jobs=-1
        )
        
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        train_acc = classifier.score(train_features, train_labels)
        val_acc = classifier.score(val_features, val_labels)
        
        results.append({
            'k': k,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        print(f"  k={k:3d}: Train={train_acc:.4f} ({train_acc*100:.2f}%), Val={val_acc:.4f} ({val_acc*100:.2f}%)")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_classifier = classifier
            best_k = k
    
    print(f"\n✓ Best k={best_k} with Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    return best_classifier, best_k


# ============================================================================
#                          SUBMISSION CREATION
# ============================================================================

def create_submission(test_features, test_filenames, classifier, output_path):
    """
    Create submission.csv for Kaggle.
    
    Args:
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        classifier: Trained KNN classifier
        output_path: Path to save submission.csv
    """
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    class_dist = submission_df['class_id'].value_counts().sort_index()
    print(f"  Unique classes predicted: {len(class_dist)}/200")
    print(f"  Most common classes:")
    print(class_dist.head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() <= 199, "Invalid class_id > 199"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")
    
    return submission_df


# ============================================================================
#                          MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Create Kaggle Submission with YOUR MoCo v3 Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to YOUR MoCo v3 checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output path for submission file')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100],
                        help='List of k values to try for KNN')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    
    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    print(f"\nCreating datasets (96×96 images, matching training)...")
    train_dataset = ImageDataset(
        data_dir / 'train',
        train_df['filename'].tolist(),
        train_df['class_id'].tolist()
    )
    
    val_dataset = ImageDataset(
        data_dir / 'val',
        val_df['filename'].tolist(),
        val_df['class_id'].tolist()
    )
    
    test_dataset = ImageDataset(
        data_dir / 'test',
        test_df['filename'].tolist(),
        labels=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize YOUR feature extractor
    feature_extractor = YourFeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        feature_extractor, train_loader, 'train'
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        feature_extractor, val_loader, 'val'
    )
    test_features, _, test_filenames = extract_features_from_dataloader(
        feature_extractor, test_loader, 'test'
    )
    
    # Train KNN classifier (with k tuning)
    classifier, best_k = train_knn_classifier(
        train_features, train_labels,
        val_features, val_labels,
        k_values=args.k_values
    )
    
    # Create submission
    submission_df = create_submission(
        test_features, test_filenames, classifier, args.output
    )
    
    print("\n" + "="*60)
    print("✓ DONE! Now upload your submission.csv to Kaggle.")
    print("="*60)
    print(f"\nYour MoCo v3 model (epoch 49) has been evaluated!")
    print(f"Best k: {best_k}")
    print(f"Submission file: {args.output}")
    print("\nGood luck with the competition! 🚀")


if __name__ == "__main__":
    main()
