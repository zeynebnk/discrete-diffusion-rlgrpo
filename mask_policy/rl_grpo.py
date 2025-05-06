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
    return (masks == y).sum(dim=-1)
 
def train_step():
    # prep! 
    G = 8  # num samples
    epsilon = 0.2  # εlip limit
    beta = 0.0005  # kl penalty weight
    lr = 0.001

    # cur and ref policy
    policy = MaskPolicy(hidden_size=512, n_layers=10, n_heads=8) 
    ref_policy = MaskPolicy(hidden_size=512, n_layers=10, n_heads=8).eval()
    ref_policy.load_state_dict(policy.state_dict())
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # pregen dataset, stepwise only!! 
    dl = DataLoader(MaskPolicyDataset(), batch_size=16, shuffle=True)
    
    for k, data_batch in enumerate(dl):
        X, y = data_batch
        gen_len = X.shape[1]

        # sample outputs from policy
        with torch.no_grad():
            fwpass = policy(X, gen_len) # bs, gen_len
            fwpass = fwpass.unsqueeze(1).expand(-1, G, -1)
            samples = torch.bernoulli(fwpass) # bs, G, gen_len

        fwpass_ref = ref_policy(X, gen_len) # bs, gen_len
        fwpass_ref = fwpass_ref.unsqueeze(1).expand(-1, G, -1)

        # Compute log probabilities
        logprobs_ref = torch.where(samples == 1, 
                       torch.log(fwpass_ref), 
                       torch.log(1 - fwpass_ref)).sum(dim=-1) # bs, G
        
        logprobs = torch.where(samples == 1, 
                       torch.log(fwpass), 
                       torch.log(1 - fwpass)).sum(dim=-1) # bs, G

        # Compute rewards and advantages
        rewards = (samples == y.unsqueeze(1).expand(-1, G, -1)).float().mean(dim=-1) + 1e-8
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-8)

        # Compute PPO ratios
        ratios = torch.exp(logprobs - logprobs_ref)
        clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

        # Compute losses according to GRPO formula
        loss_policy = -torch.min(ratios * advantages, clipped_ratios * advantages).mean() / G
        
        # Compute KL penalty according to GRPO formula
        ratio_kl = fwpass_ref.detach() / fwpass
        kl_penalty = (ratio_kl - torch.log(ratio_kl) - 1).mean()

        total_loss = loss_policy + beta * kl_penalty

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"loss: {loss_policy.item()}, kl: {kl_penalty.item()}, avg_r: {rewards.mean().item()}")


def train_loop():

    epochs = 100

    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}")
        train_step()

train_loop()




        
        
        

            
            
              


        

    


