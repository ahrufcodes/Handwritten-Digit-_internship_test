import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 50
LATENT_DIM = 20
INPUT_DIM = 784  # 28x28

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

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model initialization
model = VAE(input_dim=INPUT_DIM, hidden_dim=400, latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}')
    
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                
    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

def generate_digits_by_class(model, device, num_samples=5):
    """Generate digits for each class 0-9"""
    model.eval()
    
    # Collect latent representations for each digit class
    digit_latents = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            
            for i in range(len(labels)):
                digit = labels[i].item()
                digit_latents[digit].append(mu[i].cpu().numpy())
    
    # Calculate mean latent representation for each digit
    digit_means = {}
    for digit in range(10):
        if digit_latents[digit]:
            digit_means[digit] = np.mean(digit_latents[digit], axis=0)
    
    # Generate samples for each digit
    generated_digits = {}
    
    for digit in range(10):
        if digit in digit_means:
            samples = []
            base_mean = torch.tensor(digit_means[digit]).to(device)
            
            for _ in range(num_samples):
                # Add small random noise to create variety
                noise = torch.randn_like(base_mean) * 0.5
                z = base_mean + noise
                
                with torch.no_grad():
                    sample = model.decode(z.unsqueeze(0))
                    sample = sample.view(28, 28).cpu().numpy()
                    samples.append(sample)
            
            generated_digits[digit] = samples
    
    return generated_digits

# Training loop
if __name__ == "__main__":
    print("Starting VAE training for MNIST digit generation...")
    
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        
        # Generate and save samples every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample = torch.randn(64, LATENT_DIM).to(device)
                sample = model.decode(sample).cpu()
                
                # Save sample images
                fig, axes = plt.subplots(8, 8, figsize=(10, 10))
                for i in range(64):
                    axes[i//8, i%8].imshow(sample[i].view(28, 28), cmap='gray')
                    axes[i//8, i%8].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'vae_samples_epoch_{epoch}.png')
                plt.close()
    
    # Save the trained model
    torch.save(model.state_dict(), 'vae_mnist_model.pth')
    print("Model saved as 'vae_mnist_model.pth'")
    
    # Generate and save digit-specific latent means for better generation
    print("Computing digit-specific latent means...")
    model.eval()
    digit_latents = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            
            for i in range(len(labels)):
                digit = labels[i].item()
                digit_latents[digit].append(mu[i].cpu().numpy())
    
    # Calculate mean latent representation for each digit
    digit_means = {}
    for digit in range(10):
        if digit_latents[digit]:
            digit_means[digit] = np.mean(digit_latents[digit], axis=0).tolist()
    
    # Save digit means for use in the Streamlit app
    with open('digit_latent_means.pkl', 'wb') as f:
        pickle.dump(digit_means, f)
    print("Digit latent means saved to 'digit_latent_means.pkl'")
    
    # Generate final samples for each digit
    print("Generating final digit samples...")
    generated_digits = generate_digits_by_class(model, device)
    
    # Save generated samples
    fig, axes = plt.subplots(10, 5, figsize=(15, 30))
    for digit in range(10):
        if digit in generated_digits:
            for i, sample in enumerate(generated_digits[digit]):
                axes[digit, i].imshow(sample, cmap='gray')
                axes[digit, i].set_title(f'Digit {digit} - Sample {i+1}')
                axes[digit, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_digits_all.png')
    plt.close()
    
    print("Training completed successfully!")
    print("Files generated:")
    print("- vae_mnist_model.pth (trained model)")
    print("- digit_latent_means.pkl (digit-specific latent representations)")
    print("- generated_digits_all.png (sample generations)") 