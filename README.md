# discrete-diffusion-rlgrpo

modified lm_eval / utils and lm_eval / models / huggingface.py for custom generate + ll

unzip custom lm-eval-harness + run:
python -m lm_eval --model hf --model_args '{"pretrained": "GSAI-ML/LLaDA-8B-Instruct", "trust_remote_code": true}' --tasks "tinyGSM8k" or "gsm8k" --num_fewshot 4 --output_path gsm8k_results.json

