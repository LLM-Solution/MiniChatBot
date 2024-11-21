#!/bin/bash
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-20 10:56:00
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-21 12:33:06

# Stop on errors
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default to GPU installation
CPU_ONLY=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cpu-only) CPU_ONLY=true ;;
        *) echo -e "${RED}Unknown parameter: $1${NC}" && exit 1 ;;
    esac
    shift
done

# Function to install PyTorch and LLaMa C++
install_pytorch_llamacpp() {
  if [ "$CPU_ONLY" = true ]; then
    echo -e "${YELLOW}Installing PyTorch and llama-cpp-python for CPU...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install llama-cpp-python
  else
    if command -v nvidia-smi &>/dev/null; then
      echo -e "${YELLOW}Installing PyTorch with GPU support...${NC}"
      pip install torch torchvision torchaudio
      pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    else
      echo -e "${RED}GPU not detected. Falling back to CPU installation.${NC}"
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      pip install llama-cpp-python
    fi
  fi
}

# Function to clone repo
check_or_clone_repo() {
  REPO_URL=$1
  REPO_DIR=$2

  if [ -d "$REPO_DIR" ]; then
    echo -e "${YELLOW}Repository $REPO_DIR already exists. Pulling latest changes...${NC}"
    cd "$REPO_DIR" && git pull && cd ..
  else
    echo -e "${YELLOW}Cloning repository $REPO_URL...${NC}"
    git clone "$REPO_URL"
  fi
}

# Function to create directory
check_or_create_dir() {
  DIR=$1

  if [ -d "$DIR" ]; then
    echo -e "${YELLOW}Directory $DIR already exists. Skipping creation.${NC}"
  else
    echo -e "${YELLOW}Creating directory $DIR...${NC}"
    mkdir -p "$DIR"
  fi
}

# Update system
echo -e "${YELLOW}Updating system...${NC}"
sudo apt update
sudo apt upgrade -y

# Install basic dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt install -y python3 python3-pip python3-venv git nginx certbot python3-certbot-nginx

# Setup Python virtual environment
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"
python3 -m venv ~/venv
source ~/venv/bin/activate

# Install PyTorch and LLaMa C++
install_pytorch_llamacpp

# Clone repositories
echo -e "${YELLOW}Cloning necessary repositories...${NC}"
check_or_clone_repo "https://github.com/LLM-Solution/PyLLMSol.git" "PyLLMSol"
check_or_clone_repo "https://github.com/ggerganov/llama.cpp.git" "llama.cpp"

# Install PyLLMSol
echo -e "${YELLOW}Installing PyLLMSol and llama-cpp-python...${NC}"

# Return to main project
cd ..

# Install requirements
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create logs directory
check_or_create_dir "~/MiniChatBot/src/logs"

# Install Gunicorn
echo -e "${YELLOW}Installing Gunicorn...${NC}"
pip install gunicorn

# Prompt user for hostname
read -p "${BLUE}Enter your API hostname (e.g., api.example.com): ${NC}" HOSTNAME
if [ -z "$HOSTNAME" ]; then
  echo -e "${RED}Hostname cannot be empty. Exiting.${NC}"
  exit 1
fi

# Define the configuration file name based on the hostname
CONFIG_FILE="/etc/nginx/sites-available/$HOSTNAME"

# Replace placeholder in Nginx configuration
echo -e "${YELLOW}Configuring Nginx with hostname: ${HOSTNAME}${NC}"
sed "s/server_name api.example.com;/server_name $HOSTNAME;/" api.example.com.template > "$HOSTNAME"

# Copy and enable the Nginx configuration
echo -e "${YELLOW}Copying configuration to Nginx directory...${NC}"
sudo mv "$HOSTNAME" "$CONFIG_FILE"
sudo ln -s "$CONFIG_FILE" /etc/nginx/sites-enabled/

# Test and restart Nginx
if ! sudo nginx -t; then
    echo -e "${RED}Nginx configuration test failed. Check your configuration.${NC}"
    exit 1
fi
sudo systemctl restart nginx

# Final message
echo -e "${GREEN}Setup complete.${NC}"
echo "${BLUE}Please copy model weights and start the server as described in the documentation.${NC}"

if ! $CPU_ONLY && ! command -v nvidia-smi &>/dev/null; then
    echo -e "${YELLOW}If you plan to use GPU acceleration, please install GPU drivers and reboot your system.${NC}"
    echo -e "${YELLOW}Refer to the README for detailed instructions.${NC}"
fi
