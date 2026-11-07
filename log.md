# LFM 1.7B + NVIDIA Loop + Cyber Security + RSI Fine-Tune
# Branch: arena/438d3bde-slm-training (Active)
# Cost: 0 (Boss PC)
# Date: 2026-11-07

# Prerequisites (Boss PC):
# - Python 3.10+
# - GPU (RTX 3090/4090/A100) or CPU (Slow)
# - Internet (Free downloads)

# 1. Download Weights (Free)
# Model: LiquidAI/LFM3-1.7B-Base (Open Weight)
# URL: https://huggingface.co/LiquidAI/LFM3-1.7B-Base
# Command: git lfs install && git clone https://huggingface.co/LiquidAI/LFM3-1.7B-Base weights/LFM3-1.7B

# 2. NVIDIA Loop Transformer Code (Free)
# Architecture: 8-layer input, 20-layer shared recurrent (loops 1-4), 8-layer output
# Based on NVIDIA Dory (3.97B) logic applied to LFM3-1.7B (1.73B)
# File: code/loop_block.py (See below)

# 3. Cyber Security Data (Free)
# Sources: GHSA, KEV, ATT&CK, Phishing, HTTP reasoning
# Command: wget https://raw.githubusercontent.com/JoasASantos/Offensive-Security-AI-Models/main/data.zip && unzip data.zip -d data/

# 4. Fine-Tune Script (0 Cost on Boss PC)
# Tool: Unsloth (Free open-source) or QLoRA (Free open-source)
# Command: python train.py --model weights/LFM3-1.7B --data data/cyber --loop_count 4 --architecture code/loop_block.py

# 5. RSI Loop (Self-Evolving)
# Agent logic: Train -> Test -> Evaluate -> Generate New Data -> Repeat
# Command: while true; do python train.py ...; python evaluate.py ...; python generate.py ...; done

# Boss Decision: Stop, Change, or Wait.
# Agent waits.
