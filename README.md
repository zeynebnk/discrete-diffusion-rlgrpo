# discrete-diffusion-rlgrpo

## dw rl
generate stepwise simulation dataset

run get_logits_dataset.py

```
# python get_logits_dataset.py
```
args: 
--steps, --block_length, --gen_length, --temperature, --remasking, --split (# prompts), --out_pref


run stepwise rl (grpo)

run rl_grpo_sw.py

```
# python rl_grpo_sw.py
```
args: 
--X_path, --y_path, --G (num samples), --epsilon (clip limit), --beta (kl penalty w), --lr, --epochs, --update_ref_every, --hidden_size, --n_layers, --n_heads, out_pref

## eval
modified lm_eval / utils and lm_eval / models / huggingface.py for custom generate + ll

unzip lm-evaluation-harness.zip
cd lm-eval-harness

python -m lm_eval --model hf --model_args '{"pretrained": "GSAI-ML/LLaDA-8B-Instruct", "trust_remote_code": true}' --tasks "tinyGSM8k" or "gsm8k" --num_fewshot 4 --output_path gsm8k_results.json
