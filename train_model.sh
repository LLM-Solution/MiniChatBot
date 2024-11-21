#!/bin/bash
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-21 16:40:18
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-21 16:56:50
# Train and quantize the MiniChatBot model

set -e  # Stop on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SIZE="1B"

MODEL_URL="https://huggingface.co/meta-llama/Llama-3.2-${SIZE}-Instruct/tree/main"
MODEL_DIR="./models/Llama-3.2-${SIZE}-Instruct/"
LORA_WEIGHTS="./models/LoRA_weights_MiniChatBot-1.0-${SIZE}/"
# TRAINED_MODEL_DIR="./models/MiniChatBot-1.0-${SIZE}/"
TRAINED_MODEL_DIR="./models/MiniChatBot-${SIZE}-Instruct/"
GGUF_MODEL="./models/MiniChatBot-1.0-${SIZE}.gguf"

echo -e "${YELLOW}Starting training pipeline...${NC}"

# Step 1: Download initial model if not exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${YELLOW}Downloading Llama-3.2-${SIZE}-Instruct model...${NC}"
    git lfs install
    git clone "$MODEL_URL" "$MODEL_DIR"
else
    echo -e "${YELLOW}Model already exists in $MODEL_DIR. Skipping download.${NC}"
fi

# Step 2: Train the model
echo -e "${YELLOW}Training the model...${NC}"
python src/trainer_instruct.py --model "$MODEL_DIR"

# Step 3: Merge LoRA weights
# echo -e "${YELLOW}Merging LoRA weights...${NC}"
# python src/lora_merger.py --model "$MODEL_DIR" --lora "$LORA_WEIGHTS" --output_path "$TRAINED_MODEL_DIR"

# Step 4: Quantize the model
echo -e "${YELLOW}Quantizing the model...${NC}"
python llama.cpp/convert_hf_to_gguf.py "$TRAINED_MODEL_DIR" --outfile "$GGUF_MODEL" --outtype q8_0

echo -e "${GREEN}Training pipeline complete. Quantized model saved at ${GGUF_MODEL}.${NC}"
