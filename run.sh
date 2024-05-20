#!/bin/bash
source ./.venv/bin/activate
python app.py \
  --model_name_or_path ./models/sakura-32b-qwen2beta-v0.9-iq4xs.gguf \
  --use_gpu \
  --translate_show_progress
