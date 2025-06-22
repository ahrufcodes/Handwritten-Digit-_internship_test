#!/usr/bin/env python3
"""
Helper script to check training status and run the Streamlit app
"""

import os
import subprocess
import time

def check_files():
    """Check if required files exist"""
    required_files = {
        'vae_mnist_model.pth': 'Trained VAE model',
        'digit_latent_means.pkl': 'Digit-specific latent means'
    }
    
    status = {}
    for file, description in required_files.items():
        exists = os.path.exists(file)
        status[file] = exists
        print(f"{'✅' if exists else '❌'} {description}: {file}")
    
    return all(status.values())

def run_streamlit():
    """Run the Streamlit app"""
    print("\n🚀 Starting Streamlit app...")
    print("🌐 The app will be available at: http://localhost:8501")
    print("📱 Use Ctrl+C to stop the app")
    
    try:
        subprocess.run(['streamlit', 'run', 'streamlit_app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    print("🔢 MNIST Digit Generator - Status Check")
    print("=" * 50)
    
    # Check if files exist
    all_ready = check_files()
    
    if all_ready:
        print("\n✅ All required files are ready!")
        print("📝 You can now run the Streamlit app")
        
        run_choice = input("\n🤔 Do you want to run the app now? (y/n): ").lower().strip()
        if run_choice in ['y', 'yes']:
            run_streamlit()
        else:
            print("💡 To run manually: streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Some files are missing!")
        print("🏃‍♂️ Make sure the training script (mnist_vae_training.py) has completed.")
        print("📊 Training typically takes 20-30 minutes on a modern CPU")
        
        # Check if training is running
        try:
            result = subprocess.run(['pgrep', '-f', 'mnist_vae_training.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("🔄 Training appears to be running...")
                print("⏳ Please wait for training to complete")
            else:
                print("🎯 Run: python3 mnist_vae_training.py")
        except:
            print("🎯 Run: python3 mnist_vae_training.py")

if __name__ == "__main__":
    main() 