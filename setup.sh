#!/bin/bash
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-20 10:56:00
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-20 17:49:19

# Stop on errors
set -e

# Colors for output
GREEN="\033[0;32m"
RESET="\033[0m"

# Function to install NVIDIA drivers and CUDA (if GPU is detected)
install_gpu_drivers() {
  if lspci | grep -i nvidia; then
    echo -e "${GREEN}NVIDIA GPU detected. Installing drivers and CUDA...${RESET}"

    # # Automatically install recommended NVIDIA drivers
    # sudo apt update
    # sudo apt install -y ubuntu-drivers-common
    # sudo ubuntu-drivers autoinstall

    # # Add CUDA repository
    # echo -e "${GREEN}Adding CUDA repository...${RESET}"
    # sudo mkdir -p /etc/apt/keyrings
    # wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | sudo gpg --dearmor -o /etc/apt/keyrings/cuda-keyring.gpg
    # echo "deb [signed-by=/etc/apt/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list

    # # Install CUDA
    # sudo apt update
    # sudo apt install -y cuda

    echo -e "${GREEN}CUDA installation complete.${RESET}"
  else
    echo -e "${GREEN}No NVIDIA GPU detected. Skipping GPU drivers installation.${RESET}"
  fi
}

# Function to install PyTorch
install_pytorch() {
  if lspci | grep -i nvidia; then
    echo -e "${GREEN}Installing PyTorch with GPU support...${RESET}"
    pip install torch torchvision torchaudio
  else
    echo -e "${GREEN}Installing PyTorch for CPU...${RESET}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
}

# Update system
echo -e "${GREEN}Updating system...${RESET}"
sudo apt update
sudo apt upgrade -y

# Install basic dependencies
echo -e "${GREEN}Installing system dependencies...${RESET}"
sudo apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx

# Setup Python virtual environment
echo -e "${GREEN}Setting up Python virtual environment...${RESET}"
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install GPU drivers if applicable
install_gpu_drivers

# Install PyTorch
install_pytorch

# Clone repositories
echo -e "${GREEN}Cloning necessary repositories...${RESET}"
git clone https://github.com/abetlen/llama-cpp-python.git
git clone https://github.com/LLM-Solution/PyLLMSol.git

# Install PyLLMSol and llama-cpp-python
echo -e "${GREEN}Installing PyLLMSol and llama-cpp-python...${RESET}"
cd PyLLMSol
pip install -e .
cd ../llama-cpp-python
pip install -e .

# Return to main project
cd ..

# Install requirements
echo -e "${GREEN}Installing Python dependencies...${RESET}"
pip install -r requirements.txt

# Create logs directory
mkdir -p ~/MiniChatBot/src/logs

# Install Gunicorn
echo -e "${GREEN}Installing Gunicorn...${RESET}"
pip install gunicorn

# Prompt user for hostname
read -p "Enter your API hostname (e.g., api.example.com): " HOSTNAME
if [ -z "$HOSTNAME" ]; then
  echo "Hostname cannot be empty. Exiting."
  exit 1
fi

# Define the configuration file name based on the hostname
CONFIG_FILE="/etc/nginx/sites-available/$HOSTNAME"

# Replace placeholder in Nginx configuration
echo -e "${GREEN}Configuring Nginx with hostname: ${HOSTNAME}${RESET}"
sed "s/server_name api.example.com;/server_name $HOSTNAME;/" api.example.com.template > "$HOSTNAME"

# Copy and enable the Nginx configuration
echo -e "${GREEN}Copying configuration to Nginx directory...${RESET}"
sudo mv "$HOSTNAME" "$CONFIG_FILE"
sudo ln -s "$CONFIG_FILE" /etc/nginx/sites-enabled/

# Test and restart Nginx
sudo nginx -t
sudo systemctl restart nginx

# Final message
echo -e "${GREEN}Setup complete.${RESET}"
echo "Please copy model weights and start the server as described in the documentation."
