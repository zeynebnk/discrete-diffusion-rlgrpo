import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle

from mask_policy import MaskPolicy

class MaskPolicyDataset(Dataset):
    def __init__(self):
        with open("logits_ds_X.pkl", "rb") as f:    
            self.X = pickle.load(f)
        with open("logits_ds_y.pkl", "rb") as f:
            self.y = pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def accuracy_reward(masks, y):
    return (masks == y).sum(dim=-1) / masks.shape[-1]
 
def train_step():
    # prep! 
    G = 100  # num samples
    epsilon = 0.2  # clip limit
    beta = 0.0001  # kl penalty weight
    lr = 0.001

    
    # cur and ref policy
    hidden_size, n_layers, n_heads = 512, 8, 8
    policy = MaskPolicy(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads)
    ref_policy = MaskPolicy(hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads).eval()
    ref_policy.load_state_dict(policy.state_dict())
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # pregen dataset, stepwise only!! 
    dl = DataLoader(MaskPolicyDataset(), batch_size=32, shuffle=True)
    for k, data_batch in enumerate(dl):
        X, y = data_batch
        gen_len = X.shape[1]

        # sample outputs from policy
        fwpass = policy(X, gen_len) # bs, gen_len
        fwpass = fwpass.unsqueeze(1).expand(-1, G, -1)

        with torch.no_grad():
            samples = torch.bernoulli(fwpass) # bs, G, gen_len

            fwpass_ref = ref_policy(X, gen_len) # bs, gen_len
            fwpass_ref = fwpass_ref.unsqueeze(1).expand(-1, G, -1)

        # logprobs
        logprobs_ref = torch.where(samples == 1, 
                       torch.log(fwpass_ref), 
                       torch.log(1 - fwpass_ref)).sum(dim=-1) # bs, G
        
        logprobs = torch.where(samples == 1, 
                       torch.log(fwpass), 
                       torch.log(1 - fwpass)).sum(dim=-1) # bs, G

        # r, a
        rewards = (samples == y.unsqueeze(1).expand(-1, G, -1)).float().mean(dim=-1) + 1e-8
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)

        # ratios
        ratios = torch.exp(logprobs - logprobs_ref)
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

        # loss
        loss_policy = -torch.min(ratios * advantages, clipped_ratios * advantages).mean() / G
        
        # kl
        ratio_kl = fwpass_ref.detach() / fwpass
        kl_penalty = (ratio_kl - torch.log(ratio_kl) - 1).mean()

        total_loss = loss_policy + beta * kl_penalty

       

        total_loss.backward()

        '''
        for param in policy.parameters():
            if param.grad is not None:
                print(param.grad.norm().item())
            else:
                print("NONE")
        '''

        optimizer.step()
        optimizer.zero_grad()
        
        # ref_policy.load_state_dict(policy.state_dict()) # for updating ref policy


        print(f"loss: {loss_policy.item()}, kl: {kl_penalty.item()}, avg_r: {rewards.mean().item()}, max_r: {rewards.max().item()}")

        return policy
def train_loop():
    all_grad_history = []
    epochs = 20

    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}")
        policy = train_step()

    torch.save(policy.state_dict(), "policy.pth")
train_loop()




        
        
        

            
            
              


        

    


