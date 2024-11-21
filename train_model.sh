#!/bin/bash
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-21 16:40:18
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-21 18:41:10
# Train and quantize the MiniChatBot model

set -e  # Stop on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SIZE="1B"
ORIGINAL_MODEL="Llama-3.2"
TRAINED_MODEL="MiniChatBot-1.0"
INSTRUCT="-Instruct"
QUANT_TYPE="q8_0"

# MODEL_URL="https://huggingface.co/meta-llama/${ORIGINAL_MODEL}-${SIZE}${INSTRUCT}"
MODEL_URL="git@hf.co:meta-llama/Llama-3.2-${SIZE}${INSTRUCT}"
MODEL_DIR="./models/${ORIGINAL_MODEL}-${SIZE}${INSTRUCT}/"
LORA_WEIGHTS="./models/LoRA_${TRAINED_MODEL}-${SIZE}${INSTRUCT}/"
TRAINED_MODEL_DIR="./models/${TRAINED_MODEL}-${SIZE}${INSTRUCT}/"
GGUF_MODEL="./models/${TRAINED_MODEL}-${SIZE}${INSTRUCT}-${QUANT_TYPE}.gguf"

echo -e "${YELLOW}Starting training pipeline...${NC}"

# Create logs directory
LOG_DIR="./logs"

if [ ! -d "$LOG_DIR" ]; then
    echo -e "${YELLOW}Directory $LOG_DIR does not exist. Creating it...${NC}"
    mkdir -p "$LOG_DIR"
else
    echo -e "${GREEN}Directory $LOG_DIR already exists.${NC}"
fi

# Step 1: Download initial model if not exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Model does not exist, please downloading it manually.${NC}"
    exit 1

    # Check if git-lfs is installed
    if ! command -v git-lfs &>/dev/null; then
        echo -e "${RED}git-lfs is not installed. Installing it...${NC}"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt update && sudo apt install -y git-lfs
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install git-lfs
        else
            echo -e "${RED}Please install git-lfs manually.${NC}"
            exit 1
        fi
        git lfs install
    fi

    echo -e "${YELLOW}Downloading Llama-3.2-${SIZE}-Instruct model...${NC}"
    git lfs install
    git clone "$MODEL_URL" "$MODEL_DIR"
else
    echo -e "${YELLOW}Model already exists in $MODEL_DIR. Skipping download.${NC}"
fi

# Activate Python virtual environment
echo -e "${YELLOW}Activate Python virtual environment...${NC}"
source ~/venv/bin/activate

# Step 2: Train the model
echo -e "${YELLOW}Training the model...${NC}"
python src/trainer_instruct.py --model "$MODEL_DIR" --lora_weights "$LORA_WEIGHTS"

# Step 3: Merge LoRA weights
echo -e "${YELLOW}Merging LoRA weights...${NC}"
python src/lora_merger.py --model "$MODEL_DIR" --lora "$LORA_WEIGHTS" --output_path "$TRAINED_MODEL_DIR"

# Step 4: Quantize the model
echo -e "${YELLOW}Quantizing the model...${NC}"
python llama.cpp/convert_hf_to_gguf.py "$TRAINED_MODEL_DIR" --outfile "$GGUF_MODEL" --outtype "$QUANT_TYPE"

echo -e "${GREEN}Training pipeline complete. Quantized model saved at ${GGUF_MODEL}.${NC}"
