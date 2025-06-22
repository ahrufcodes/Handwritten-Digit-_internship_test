#!/usr/bin/env python3
"""
Generate digit-specific latent means from the trained VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pickle

# VAE Model Definition (same as training script)
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar layer
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def generate_digit_means():
    """Generate digit-specific latent means from trained model"""
    
    print("🔄 Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the trained model
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    model.load_state_dict(torch.load('vae_mnist_model.pth', map_location=device))
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Load MNIST training data
    print("📊 Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    
    # Collect latent representations for each digit class
    print("🧮 Computing latent representations for each digit...")
    digit_latents = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            
            for i in range(len(labels)):
                digit = labels[i].item()
                digit_latents[digit].append(mu[i].cpu().numpy())
            
            # Progress indicator
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}/{len(train_loader)}")
    
    # Calculate mean latent representation for each digit
    print("📈 Calculating mean representations...")
    digit_means = {}
    for digit in range(10):
        if digit_latents[digit]:
            mean_latent = np.mean(digit_latents[digit], axis=0)
            digit_means[digit] = mean_latent.tolist()
            print(f"Digit {digit}: {len(digit_latents[digit])} samples")
    
    # Save digit means
    with open('digit_latent_means.pkl', 'wb') as f:
        pickle.dump(digit_means, f)
    
    print("💾 Digit latent means saved to 'digit_latent_means.pkl'")
    
    # Test generation for each digit
    print("🎨 Testing generation for each digit...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Sample Generated Digits (One per class)', fontsize=16)
    
    for digit in range(10):
        if digit in digit_means:
            # Generate one sample for this digit
            base_mean = torch.tensor(digit_means[digit], dtype=torch.float32).to(device)
            noise = torch.randn_like(base_mean) * 0.2  # Small noise for variety
            z = base_mean + noise
            
            with torch.no_grad():
                sample = model.decode(z.unsqueeze(0))
                sample = sample.view(28, 28).cpu().numpy()
            
            row = digit // 5
            col = digit % 5
            axes[row, col].imshow(sample, cmap='gray')
            axes[row, col].set_title(f'Digit {digit}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_digit_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("🖼️  Test samples saved to 'test_digit_samples.png'")
    print("✅ All done! You can now run the Streamlit app.")

if __name__ == "__main__":
    generate_digit_means() 