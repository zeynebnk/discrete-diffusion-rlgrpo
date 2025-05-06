# discrete-diffusion-rlgrpo

## rl
get logits_ds_X.pkl, logits_ds_y.pkl (or run get_logits_dataset.py) -> run rl_grpo.py

## eval
modified lm_eval / utils and lm_eval / models / huggingface.py for custom generate + ll

unzip lm-evaluation-harness.zip
cd lm-eval-harness

python -m lm_eval --model hf --model_args '{"pretrained": "GSAI-ML/LLaDA-8B-Instruct", "trust_remote_code": true}' --tasks "tinyGSM8k" or "gsm8k" --num_fewshot 4 --output_path gsm8k_results.json
