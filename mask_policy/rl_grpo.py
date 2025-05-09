import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle

from mask_policy import MaskPolicy


class MaskPolicyDataset(Dataset):
    def __init__(self):
        with open("logits_ds_X (1).pkl", "rb") as f:    
            self.X = pickle.load(f)
        with open("logits_ds_y (2).pkl", "rb") as f:
            self.y = pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DummyDataset(Dataset):
    def __init__(self):
        self.X = torch.rand((100,5,256))
        self.y = (self.X.sum(dim=2) > 128) * 1
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def kl(p, p_ref):
    
    # get kl / probs
    kl_pos = p * (torch.log(p) - torch.log(p_ref))
    kl_neg = (1 - p) * (torch.log(1 - p) - torch.log(1 - p_ref))
    
    kl = kl_pos + kl_neg
    
    return kl.mean()

def train_step(policy, ref_policy, optimizer, G, epsilon, beta):

    # pregen dataset, stepwise only!! 
    dl = DataLoader(DummyDataset(), batch_size=16, shuffle=True)
    for k, data_batch in enumerate(dl):
        X, y = data_batch
        gen_len = X.shape[1]

        # sample outputs from policy
        fwpass = policy(X) # bs, gen_len
        fwpass = fwpass.unsqueeze(1).expand(-1, G, -1)
        fwpass = fwpass.clamp(1e-8, 1.0 - 1e-8)
        
        with torch.no_grad():

            fwpass_ref = ref_policy(X) # bs, gen_len
            fwpass_ref = fwpass_ref.unsqueeze(1).expand(-1, G, -1)

            samples = torch.bernoulli(fwpass) # bs, G, gen_len
        
        fwpass_ref = fwpass_ref.clamp(1e-8, 1.0 - 1e-8)
           

        log_probs = torch.where(samples == 1, 
                       torch.log(fwpass), 
                       torch.log(1 - fwpass))

        log_probs_ref = torch.where(samples == 1, 
                                torch.log(fwpass_ref), 
                                torch.log(1 - fwpass_ref))

        ratios = torch.exp(log_probs.sum(dim=-1) - log_probs_ref.sum(dim=-1).detach())
        
        # r, a
        rewards = (samples == y.unsqueeze(1).expand(-1, G, -1)).float().mean(dim=-1)
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

def train_loop():
    # prep! 
    update_ref_every = 10
        
    G = 4  # num samples
    epsilon = 0.2  # clip limit
    beta = 0.0001  # kl penalty weight
    lr = 1e-5

    # cur and ref policy
    hidden_size, n_layers, n_heads = 256, 4, 8
    policy = MaskPolicy(2,hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads)
    ref_policy = MaskPolicy(2,hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads).eval()
    ref_policy.load_state_dict(policy.state_dict())
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    epochs = 500

    '''with open("logits_ds_X (1).pkl", "rb") as f:    
        fullX = pickle.load(f)
    with open("logits_ds_y (2).pkl", "rb") as f:
        fully = pickle.load(f)'''

    fullX = DummyDataset().X
    fully = DummyDataset().y
    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}")
        policy = train_step(policy, ref_policy, optimizer, G, epsilon, beta)

        with torch.no_grad():
            pred = (policy(fullX) > 0.5) * 1
            print((pred == fully).float().mean(dim=-1).mean())

        if i % update_ref_every == 0:
            ref_policy.load_state_dict(policy.state_dict()) # for updating ref policy
        
    
    torch.save(policy.state_dict(), "policy.pth")

    return policy

train_loop()




        
        
        

            
            
              


        

    


