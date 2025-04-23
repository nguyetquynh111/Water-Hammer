import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the DenseNet121 architecture
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, dropout_rate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_dense_layer(in_channels + i * growth_rate, growth_rate, dropout_rate))
    
    def _make_dense_layer(self, in_channels, growth_rate, dropout_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, in_channels=1, growth_rate=32, block_config=(6, 12, 24, 16), dropout_rate=0.0):
        super(DenseNet121, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_channels = 64
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers, dropout_rate)
            self.dense_blocks.append(block)
            num_channels += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2
        
        # Final batch norm
        self.final_bn = nn.BatchNorm2d(num_channels)
        self.final_relu = nn.ReLU(inplace=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_channels, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x = self.features(x)
        
        # Dense blocks
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transitions[i](x)
        
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x

# Custom dataset class
class VesselDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Data loading function
def load_data(data_dir, img_size=(512, 512)):
    image_dir = os.path.join(data_dir, 'X')
    mask_dir = os.path.join(data_dir, 'y')
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
    
    # Sort to ensure matching pairs
    image_paths.sort()
    mask_paths.sort()
    
    # Split into train and validation sets
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = VesselDataset(train_img_paths, train_mask_paths, transform)
    val_dataset = VesselDataset(val_img_paths, val_mask_paths, transform)
    
    return train_dataset, val_dataset

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    best_f1_score = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Initialize metrics accumulators for training
        train_total_pixels = 0
        train_correct_pixels = 0
        train_true_positives = 0
        train_false_positives = 0
        train_false_negatives = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate metrics using the same predictions
            predictions = (outputs > 0.5).float()
            
            # Accumulate metrics
            train_total_pixels += masks.numel()
            train_correct_pixels += (predictions == masks).sum().item()
            train_true_positives += ((predictions == 1) & (masks == 1)).sum().item()
            train_false_positives += ((predictions == 1) & (masks == 0)).sum().item()
            train_false_negatives += ((predictions == 0) & (masks == 1)).sum().item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calculate training metrics
        train_pixel_accuracy = train_correct_pixels / train_total_pixels
        
        if train_true_positives + train_false_positives == 0:
            train_precision = 0
        else:
            train_precision = train_true_positives / (train_true_positives + train_false_positives)
        
        if train_true_positives + train_false_negatives == 0:
            train_recall = 0
        else:
            train_recall = train_true_positives / (train_true_positives + train_false_negatives)
        
        if train_precision + train_recall == 0:
            train_f1 = 0
        else:
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall)
        
        train_metrics = {
            'pixel_accuracy': train_pixel_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1
        }
        train_metrics_history.append(train_metrics)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Initialize metrics accumulators for validation
        val_total_pixels = 0
        val_correct_pixels = 0
        val_true_positives = 0
        val_false_positives = 0
        val_false_negatives = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # Calculate metrics using the same predictions
                predictions = (outputs > 0.5).float()
                
                # Accumulate metrics
                val_total_pixels += masks.numel()
                val_correct_pixels += (predictions == masks).sum().item()
                val_true_positives += ((predictions == 1) & (masks == 1)).sum().item()
                val_false_positives += ((predictions == 1) & (masks == 0)).sum().item()
                val_false_negatives += ((predictions == 0) & (masks == 1)).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        val_pixel_accuracy = val_correct_pixels / val_total_pixels
        
        if val_true_positives + val_false_positives == 0:
            val_precision = 0
        else:
            val_precision = val_true_positives / (val_true_positives + val_false_positives)
        
        if val_true_positives + val_false_negatives == 0:
            val_recall = 0
        else:
            val_recall = val_true_positives / (val_true_positives + val_false_negatives)
        
        if val_precision + val_recall == 0:
            val_f1 = 0
        else:
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
        
        val_metrics = {
            'pixel_accuracy': val_pixel_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': val_f1
        }
        val_metrics_history.append(val_metrics)
        
        # Save best model based on F1 score
        if val_metrics['f1_score'] > best_f1_score:
            best_f1_score = val_metrics['f1_score']
            # Save the entire model using torch.jit
            model.eval()  # Set to eval mode before saving
            traced_model = torch.jit.trace(model, torch.randn(1, 1, 512, 512).to(device))
            torch.jit.save(traced_model, 'best_model.pth')
            model.train()  # Set back to train mode
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Metrics - Accuracy: {train_metrics["pixel_accuracy"]:.4f}, Precision: {train_metrics["precision"]:.4f}, Recall: {train_metrics["recall"]:.4f}, F1: {train_metrics["f1_score"]:.4f}')
        print(f'Val Metrics - Accuracy: {val_metrics["pixel_accuracy"]:.4f}, Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1_score"]:.4f}')
    
    return train_losses, val_losses, train_metrics_history, val_metrics_history

# Evaluation metrics
def calculate_metrics(model, data_loader, device):
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = (outputs > 0.5).float()
            
            # Calculate metrics
            total_pixels += masks.numel()
            correct_pixels += (predictions == masks).sum().item()
            
            true_positives += ((predictions == 1) & (masks == 1)).sum().item()
            false_positives += ((predictions == 1) & (masks == 0)).sum().item()
            false_negatives += ((predictions == 0) & (masks == 1)).sum().item()
    
    # Calculate metrics
    pixel_accuracy = correct_pixels / total_pixels
    
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)
    
    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Prediction function
def predict_image(model, image_path, device, img_size=(512, 512)):
    # Load and preprocess image
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output > 0.5).float()
    
    # Convert to numpy array
    prediction = prediction.squeeze().cpu().numpy()
    
    return prediction

# Visualization function
def visualize_prediction(image_path, prediction, save_path=None):
    # Load original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (512, 512))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display prediction
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Main function
def main():
    # Set paths
    data_dir = 'SOE/train_data'  # Update with your dataset path
    
    # Load data
    train_dataset, val_dataset = load_data(data_dir)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = DenseNet121().to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train model
    train_losses, val_losses, train_metrics_history, val_metrics_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=60, device=device
    )
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot pixel accuracy
    plt.subplot(2, 2, 2)
    plt.plot([m['pixel_accuracy'] for m in train_metrics_history], label='Train Accuracy')
    plt.plot([m['pixel_accuracy'] for m in val_metrics_history], label='Validation Accuracy')
    plt.title('Pixel Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(2, 2, 3)
    plt.plot([m['precision'] for m in train_metrics_history], label='Train Precision')
    plt.plot([m['precision'] for m in val_metrics_history], label='Validation Precision')
    plt.plot([m['recall'] for m in train_metrics_history], label='Train Recall')
    plt.plot([m['recall'] for m in val_metrics_history], label='Validation Recall')
    plt.title('Precision and Recall History')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot([m['f1_score'] for m in train_metrics_history], label='Train F1')
    plt.plot([m['f1_score'] for m in val_metrics_history], label='Validation F1')
    plt.title('F1 Score History')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Load best model for prediction
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation on validation set
    final_metrics = calculate_metrics(model, val_loader, device)
    print("\nFinal Evaluation Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Example prediction
    test_image_path = '/home/quynhng/vessel/SOE/test_data/X/BAU_DO_THIse000_74.jpg'  # Update with your test image path
    prediction = predict_image(model, test_image_path, device)
    visualize_prediction(test_image_path, prediction, 'prediction_result.png')

if __name__ == "__main__":
    main() 