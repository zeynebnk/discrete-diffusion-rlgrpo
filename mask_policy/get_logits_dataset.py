import torch
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import AutoTokenizer, AutoModel

from llada_utils import add_gumbel_noise, get_num_transfer_tokens
from gsm_utils import get_gsm8k_questions

import argparse

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)  
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True).to('cuda')
    return tokenizer, model

def generate_dataset(args):
    tokenizer, model = load_model()

    prompts = get_gsm8k_questions(args.split)['question']
    print(prompts)
    
    agg_X, agg_y = [], []
    for prompt in prompts:
        print(prompt)

        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)

        
        _, agg_X, agg_y = generate_for_datagen(agg_X, agg_y, model, input_ids, steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=args.temperature, remasking=args.remasking)
    
    agg_X = torch.cat(agg_X, dim=0).cpu()
    agg_y = torch.cat(agg_y, dim=0).cpu()
    
    with open(args.out_pref+"_ds_X.pkl", "wb") as f:
        pickle.dump(agg_X, f)
    with open(args.out_pref+"_ds_y.pkl", "wb") as f:
        pickle.dump(agg_y, f)

def create_datapoint(agg_X, agg_y, logits, remasked_idx, timestep):
    if not isinstance(agg_X, list):
        agg_X = [agg_X]
    if not isinstance(agg_y, list):
        agg_y = [agg_y]
    
    agg_X.append(logits)
    agg_y.append(remasked_idx)
    
    return agg_X, agg_y


@ torch.no_grad()
def generate_for_datagen(agg_X, agg_y, model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                out = model(x, output_hidden_states=True)
                logits = out.logits
                hidden_states = out.hidden_states[-1]
                

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            agg_X, agg_y = create_datapoint(agg_X, agg_y, hidden_states[:, prompt.shape[1]:, :].cpu(), ~(x == mask_index)[:,prompt.shape[1]:].cpu(), i)
            print(len(agg_X))
            
    return x, agg_X, agg_y


def parse_args():
    parser = argparse.ArgumentParser(description='Generate stepwise dataset')


    parser.add_argument('--steps', type=int, default=64,
                      help='Number of steps')
    parser.add_argument('--block_length', type=int, default=128,
                      help='Block length for generation')
    parser.add_argument('--gen_length', type=int, default=128,
                      help='Generation length') 
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Sampling temperature')
    parser.add_argument('--remasking', type=str, default='low_confidence',
                      choices=['random', 'low_confidence', 'mask_policy'],
                      help='Remasking strategy to use')
    
    parser.add_argument('--split', type=str, default="train[:5]",
                      help='Split to use')
    parser.add_argument('--out_pref', type=str, default="logits_ds",
                      help='Prefix for output files')
    
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args)
