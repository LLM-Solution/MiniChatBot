#!/bin/bash
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-21 11:30:24
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-21 11:33:35

# Define colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run this script as root.${NC}"
  exit 1
fi

# Remove existing CUDA and NVIDIA drivers
echo -e "${YELLOW}Removing existing NVIDIA and CUDA drivers...${NC}"
apt --purge remove "*cud*" "*nvidia*" -y

# Update and upgrade system packages
echo -e "${YELLOW}Updating system packages...${NC}"
apt update && apt upgrade -y

# Install Ubuntu drivers common tools
echo -e "${YELLOW}Installing ubuntu-drivers-common...${NC}"
apt install -y ubuntu-drivers-common

# List available drivers
echo -e "${BLUE}Detecting available NVIDIA drivers...${NC}"
ubuntu-drivers devices

# Autoinstall recommended drivers
echo -e "${YELLOW}Installing recommended NVIDIA drivers...${NC}"
ubuntu-drivers autoinstall

# Verify installation
echo -e "${YELLOW}Verifying NVIDIA driver installation...${NC}"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi
  echo -e "${GREEN}NVIDIA drivers installed successfully.${NC}"
  echo -e "${BLUE}Please reboot your system to apply changes.${NC}"
else
  echo -e "${RED}NVIDIA drivers installation failed. Please check logs for details.${NC}"
fi
