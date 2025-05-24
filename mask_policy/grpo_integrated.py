import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transformers import AutoTokenizer, AutoModel

from mask_policy import MaskPolicy_LLaDA
# from rewards import get_reward

from llada_utils import add_gumbel_noise, get_num_transfer_tokens
from gsm_utils import get_gsm8k_questions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)  
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True).to('cuda')
    return tokenizer, model

def get_reward(trajs):
    return torch.randn(len(trajs)).view(-1, 2).to(device)

def train():
    ## SET GRPO PARAMS
    G = 2
    epsilon = 0.2
    beta = 0.01
    lr = 0.0001
    epochs = 10
    grpo_temp = 0.2

    tokenizer, model = load_model()

    # cur and ref policy
    policy = MaskPolicy_LLaDA().to(device)
    ref_policy = MaskPolicy_LLaDA().to(device).eval()
    ref_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    prompts = get_gsm8k_questions('train[:1]')['question']
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}, ], add_generation_prompt=True, tokenize=False) for prompt in prompts]
    
    data = list(zip([prompts[0]]*64, range(64))) 

    for _ in range(epochs):
        train_step(policy, ref_policy, optimizer, G, epsilon, beta, model, tokenizer, data, grpo_temp)
    

def kl(p, p_ref):
    
    # get kl / probs
    kl_pos = p * (torch.log(p) - torch.log(p_ref))
    kl_neg = (1 - p) * (torch.log(1 - p) - torch.log(1 - p_ref))
    
    kl = kl_pos + kl_neg
    
    return kl.mean()


def train_step(policy, ref_policy, optimizer, G, epsilon, beta, model, tokenizer, data, grpo_temp):
   
    for X, target_step in data:
        print("step")
        input_ids = tokenizer(X)['input_ids']
        input_ids = torch.tensor(input_ids).to(model.device).unsqueeze(0)
        
        gen_len = input_ids.shape[1]
        
        # sample outputs from policy
        ## GEN LEN / STEPS SET HERE
        trajs, fwpass, samples, rout = rollouts(model, policy, target_step, input_ids, 
                                 G, grpo_temp, restrict = True, steps=64, gen_length=128)
        print("got rollouts")
        with torch.no_grad():
            fwpass_ref = ref_policy(rout)
            fwpass_ref = fwpass_ref.unsqueeze(1).expand(-1, G, -1) 
            fwpass_ref = fwpass_ref.clamp(1e-8, 1.0 - 1e-8)

        fwpass = fwpass.requires_grad_(True)
        
        log_probs = torch.where(samples == 1, 
                       torch.log(fwpass), 
                       torch.log(1 - fwpass))

        log_probs_ref = torch.where(samples == 1, 
                                torch.log(fwpass_ref), 
                                torch.log(1 - fwpass_ref))

        ratios = torch.exp(log_probs.sum(dim=-1) - log_probs_ref.sum(dim=-1).detach())
        
        # r, a
        rewards = get_reward(trajs)
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)
        
        # loss
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
        loss_policy = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
        
        # kl
        kl_penalty = kl(fwpass, fwpass_ref.detach())

        total_loss = loss_policy + beta * kl_penalty

        optimizer.zero_grad()
        total_loss.backward()
        
        
        '''for param in policy.parameters():
            if param.grad is not None:
                print(param.grad.norm().item())
            else:
                print("NONE")'''
    
        optimizer.step()


        print(f"loss: {loss_policy.item()}, kl: {kl_penalty.item()}, avg_r: {rewards.mean().item()}, max_r: {rewards.max().item()}")

    return policy


@ torch.no_grad()
def rollouts(model, mask_policy, target_step, prompt, G, grpo_temp, restrict = True, steps=64, gen_length=128, mask_id=126336):
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    
    mask_index = (x == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
    
    for i in range(steps):
        print(i)
        mask_index = (x == mask_id)
        
        out = model(x, output_hidden_states=True)
        logits = out.logits
        hidden_states = out.hidden_states[-1]
        
        x0 = torch.argmax(logits, dim=-1) # b, l

        if i < target_step:
            # use mask policy! (pre target step)
            x0_p = 1 - mask_policy(hidden_states[:, prompt.shape[1]:, :].to(torch.float32))
            ones = torch.ones((x0_p.shape[0], prompt.shape[1]), device=x0_p.device)
            x0_p = torch.cat([ones, x0_p], dim=1)

            # disable remasking
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # select topk, get next input 
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        elif i == target_step:
            with torch.enable_grad():
                fwpass = mask_policy(hidden_states[:, prompt.shape[1]:, :].to(torch.float32)) # bs, gen_len
            
                fwpass = fwpass.clamp(1e-8, 1.0 - 1e-8)
                fwpass = fwpass.unsqueeze(1).expand(-1, G, -1) # bs, G, gen_len
            fwpass_noised = add_gumbel_noise(fwpass, temperature=grpo_temp)

            xs = []
            idxs = []
            for g in range(G):
                ones = torch.ones((fwpass.shape[0], prompt.shape[1]), device=fwpass_noised.device)
                x0_p = 1 - fwpass_noised[:, g]  
                x0_p = torch.cat([ones, x0_p], dim=1)

                # disable remasking
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                
                x_sample = x.clone()
                x_sample[transfer_index] = x0[transfer_index]
                xs.append(x_sample)
                idxs.append(transfer_index.clone().float()[:, prompt.shape[1]:])
            
            idxs = torch.stack(idxs, dim=1)  # bs, G, gen_len
            rout = hidden_states[:, prompt.shape[1]:, :].to(torch.float32)
            
        elif i > target_step:
            # use deterministic low confidence (post target step)
            trajs = []
            for j, x_sample in enumerate(xs):
                traj = rollouts_post(model, x_sample, num_transfer_tokens, steps-target_step, gen_length, mask_id)
                trajs.append(traj)
    return trajs, fwpass, idxs, rout


@ torch.no_grad()
def rollouts_post(model, x, num_transfer_tokens, steps=128, gen_length=128, mask_id=126336):
    traj = []
    for i in range(steps):
        mask_index = (x == mask_id)
            
        out = model(x)
        logits = out.logits

        x0 = torch.argmax(logits, dim=-1) # b, l    
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x_step = x.clone()
        x_step[transfer_index] = x0[transfer_index]
        
        traj.append(x_step)
    return traj


def main():
    train()

if __name__ == "__main__":
    main()
