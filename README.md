# MiniChatBot Backend for LLM Solutions Website

MiniChatBot is a lightweight and efficient chatbot designed to run on a web platform. This repository contains scripts to retrain models, the backend setup, including deployment scripts, dependency management, and configuration for GPU and CPU environments.

## Features

### General

- **Simplified Setup**: Easily configure the backend for both GPU and CPU environments using a `Makefile` or `setup.sh`.
- **Environment Isolation**: Includes Python virtual environment setup to ensure dependency management and avoid conflicts.
- **Dynamic GPU Support**: Automatically detects GPU availability and installs appropriate dependencies. Optional GPU driver installation instructions are provided.

### Training

- **Custom Model Fine-Tuning**: Tailor the chatbot's behavior by training or fine-tuning models using the `transformers` library.
- **Efficient Training Process**: Leverage existing scripts and tools to streamline model customization.

### Inference

- **Lightweight Integration**: Seamless integration with llama.cpp for efficient and fast inference, even in resource-constrained environments.
- **Secure API Hosting**: Pre-configured Nginx setup for reliable and secure API hosting.

## Installation

### 1. System Requirements

Ensure your system meets the following prerequisites:
- **Operating System**: Ubuntu 24.04 or later
- **Python**: Version 3.12 or higher
- **GPU Support** (Optional): NVIDIA GPU with CUDA

**Tip**: Run `python3 --version` and `nvidia-smi` to check Python and GPU availability.

### 2. Clone the Repository

Clone the MiniChatBot repository:

```bash
git clone https://github.com/LLM-Solution/MiniChatBot.git
cd MiniChatBot
```

### 3. Install GPU Drivers (Optional)

If you plan to use GPU acceleration, install GPU drivers, use the Makefile:

```bash
make install_gpu
```

**Important**: A system reboot is required after installing GPU drivers.

### 4. Setup Backend

Run the following command to set up the backend using the Makefile:

```bash
make setup
```

The script will:
- Update your system and install dependencies.
- Set up a Python virtual environment and install required Python libraries.
- Detect GPU availability and install the appropriate PyTorch version.
- Clone necessary repositories (e.g., `llama.cpp`, `PyLLMSol`).
- Configure Nginx for your API.

### 5. Start the API

Once the setup is complete, start the backend server using:

```bash
make run
```

## Makefile Commands

| Command | Description |
| --- | --- |
| `make setup` | Complete backend setup, including dependencies and Nginx setup. |
| `make install_gpu` | Install NVIDIA GPU drivers (optional). |
| `make run` | Start the backend API server using Gunicorn. |
| `make status` | Check backend server status. |
| `make stop` | Stop the backend server. |
| `make update` | Pull latest code and update dependencies. |
| `make clean` | Clean up logs and temporary files. |”

## Configuration

### 1. Nginx Configuration

During the setup, you’ll be prompted to provide your API’s hostname (e.g., `api.example.com`). This hostname will be used in the Nginx configuration.

To manually edit or check the configuration:

- Configuration file: `/etc/nginx/sites-available/<hostname>`
- Enable the configuration: sudo ln -s `/etc/nginx/sites-available/<hostname> /etc/nginx/sites-enabled/`
- Test and restart Nginx:

```bash
sudo nginx -t
sudo systemctl restart nginx
```

### 2. API Hostname

If you need to change the hostname later, edit the Nginx configuration and restart the server.

## Directory Structure

```
MiniChatBot/
├── data/                  # Data files for training the model
├── model/                 # Directory for trained models
├── Prompts/               # Initial prompts for the chatbot
├── install_gpu.sh         # Optional script for GPU driver installation
├── setup.sh               # Main setup script
├── train_model.sh         # Script to train, merge, and quantize the LLM
├── Makefile               # Makefile for common tasks
├── requirements.txt       # Python dependencies
├── src/                   # Source code directory
│   ├── logs/              # Directory for log files
│   ├── api.py             # Script to connect MiniChatBot to the API
│   ├── config.py          # Configuration file
│   ├── lora_merger.py     # Script to merge LoRA weights with the base model
│   ├── make_full_data.py  # Script to gather and prepare data
│   ├── trainer.py         # Script to retrain the LLM
│   ├── utils.py           # Utility functions
│   ├── wsgi.py            # WSGI entry point
│   └── tests/             # Test module for unit and integration tests
│       └── ...
├── PyLLMSol/              # Cloned repository for PyLLMSol
├── llama.cpp/             # Cloned repository for llama.cpp
└── README.md              # Project documentation

```

## Common Issues

### 1. GPU Not Detected

If `nvidia-smi` is not available:
- Ensure NVIDIA drivers are installed by running:
```bash
sudo make install_gpu
```
- Reboot the system and verify with:
```bash
nvidia-smi
```

Otherwise you can also run model on CPU only:
```bash
./setup.sh --cpu-only
```

### 2. Dependency Conflicts

If you encounter Python dependency issues, remove the venv directory and restart the setup:

```bash
rm -rf ~/venv
sudo bash setup.sh
```

## Contributing

We welcome contributions! Feel free to fork this repository and submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
