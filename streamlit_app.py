import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Generator", 
    page_icon="🔢",
    layout="wide"
)

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

@st.cache_resource
def load_model():
    """Load the pre-trained VAE model"""
    device = torch.device('cpu')  # Use CPU for Streamlit deployment
    model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    
    try:
        # Try to load the model state dict
        model.load_state_dict(torch.load('vae_mnist_model.pth', map_location=device))
        model.eval()
        return model, True
    except FileNotFoundError:
        st.error("Model file 'vae_mnist_model.pth' not found. Please train the model first.")
        return model, False

@st.cache_data
def load_digit_means():
    """Load digit-specific latent means if available"""
    try:
        with open('digit_latent_means.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("Digit latent means not found. Using fallback values.")
        return None

@st.cache_data
def generate_digit_samples(selected_digit, num_samples=5):
    """Generate samples for a specific digit"""
    model, model_loaded = load_model()
    
    if not model_loaded:
        return None
    
    device = torch.device('cpu')
    
    # Load actual digit-specific latent means from training
    digit_means = load_digit_means()
    
    if digit_means and selected_digit in digit_means:
        # Use the actual trained latent means
        base_mean = torch.tensor(digit_means[selected_digit], dtype=torch.float32).to(device)
    else:
        # Fallback: use random latent vector if means not available
        base_mean = torch.randn(20).to(device)
        st.warning(f"Using random latent vector for digit {selected_digit}. For better results, ensure the model was trained with the updated script.")
    
    samples = []
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            # Add random noise for variety
            noise_scale = 0.3 + (i * 0.1)  # Varying noise for different samples
            noise = torch.randn_like(base_mean) * noise_scale
            z = base_mean + noise
            
            # Generate image
            generated = model.decode(z.unsqueeze(0))
            image = generated.view(28, 28).cpu().numpy()
            
            # Ensure values are in [0, 1] range
            image = np.clip(image, 0, 1)
            samples.append(image)
    
    return samples

def create_image_grid(images, digit):
    """Create a grid of generated images"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(f'Generated Handwritten Digit: {digit}', fontsize=16, fontweight='bold')
    
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray', interpolation='nearest')
        axes[i].set_title(f'Sample {i+1}', fontsize=12)
        axes[i].axis('off')
        
        # Add border to make it look more like MNIST
        axes[i].add_patch(plt.Rectangle((0, 0), 27, 27, fill=False, edgecolor='black', lw=1))
    
    plt.tight_layout()
    
    # Convert to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    return buf

# Streamlit App
def main():
    st.title("🔢 MNIST Handwritten Digit Generator")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to the Handwritten Digit Generator!
    
    This app uses a **Variational Autoencoder (VAE)** trained on the MNIST dataset to generate 
    handwritten digit images. Select a digit below and click generate to see 5 unique variations!
    
    **Features:**
    - 🎯 Generate digits 0-9 on demand
    - 🎨 Each generation produces 5 unique variations
    - 📊 28x28 grayscale images similar to MNIST format
    - 🤖 Powered by PyTorch VAE model trained from scratch
    """)
    
    # Model status
    model, model_loaded = load_model()
    
    if model_loaded:
        st.success("✅ VAE Model loaded successfully!")
    else:
        st.error("❌ VAE Model not found. Please ensure 'vae_mnist_model.pth' is in the app directory.")
        st.info("To train the model, run the `mnist_vae_training.py` script first.")
        return
    
    st.markdown("---")
    
    # Digit selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select Digit")
        selected_digit = st.selectbox(
            "Choose a digit to generate:",
            options=list(range(10)),
            index=0,
            help="Select any digit from 0 to 9"
        )
        
        generate_button = st.button(
            "🎲 Generate 5 Samples", 
            type="primary",
            help="Click to generate 5 unique variations of the selected digit"
        )
    
    with col2:
        st.subheader("Generated Images")
        
        if generate_button:
            with st.spinner(f"Generating handwritten digit {selected_digit}..."):
                # Generate samples
                samples = generate_digit_samples(selected_digit, num_samples=5)
                
                if samples is not None:
                    # Create and display image grid
                    image_buf = create_image_grid(samples, selected_digit)
                    st.image(image_buf, use_column_width=True)
                    
                    st.success(f"✨ Successfully generated 5 variations of digit **{selected_digit}**!")
                    
                    # Additional info
                    st.info("""
                    **About the generation:**
                    - Each sample is generated with slight random variations
                    - Images are 28x28 pixels in grayscale format
                    - The VAE model learned to generate digits from MNIST dataset patterns
                    """)
                else:
                    st.error("Failed to generate samples. Please check the model.")
        else:
            st.info("👆 Select a digit and click 'Generate 5 Samples' to see the magic!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Technical Details:**
    - Model: Variational Autoencoder (VAE) with 20-dimensional latent space
    - Framework: PyTorch
    - Dataset: MNIST (28x28 grayscale handwritten digits)
    - Training: From scratch on Google Colab with T4 GPU
    
    **Created for Problem 3 - Handwritten Digit Generation Challenge**
    """)

if __name__ == "__main__":
    main() 