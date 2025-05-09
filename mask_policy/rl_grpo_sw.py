import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle

from mask_policy import MaskPolicy

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskPolicyDataset(Dataset):
    def __init__(self, X_path, y_path):
        with open(X_path, "rb") as f:    
            self.X = pickle.load(f)
        with open(y_path, "rb") as f:
            self.y = pickle.load(f)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DummyDataset(Dataset):
    def __init__(self):
        self.X = torch.rand((500,5,256))
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

def train_step(policy, ref_policy, optimizer, G, epsilon, beta, dl):
   
    for k, data_batch in enumerate(dl):
        X, y = data_batch
        X, y = X.to(device), y.to(device)  
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

def train_loop(args): 
    # prep! 
    update_ref_every = args.update_ref_every
        
    G = args.G  # num samples
    epsilon = args.epsilon  # clip limit
    beta = args.beta  # kl penalty weight
    lr = args.lr

    # cur and ref policy
    hidden_size, n_layers, n_heads = args.hidden_size, args.n_layers, args.n_heads
    policy = MaskPolicy(126464,hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads).to(device)
    ref_policy = MaskPolicy(126464,hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads).to(device).eval()
    ref_policy.load_state_dict(policy.state_dict())
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # pregen dataset, stepwise only!! 
    ds = MaskPolicyDataset(args.X_path, args.y_path)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    epochs = args.epochs

    hist = []


    for i in range(epochs):
        print(f"Epoch {i+1} of {epochs}")
        policy = train_step(policy, ref_policy, optimizer, G, epsilon, beta, dl)

        with torch.no_grad():
            pred = (policy(ds.X) > 0.5) * 1
            acc = (pred == ds.y).float().mean(dim=-1).mean()
            hist.append(acc.item())

            print(acc)

        if i % update_ref_every == 0:
           ref_policy.load_state_dict(policy.state_dict()) # for updating ref policy
        
    
    torch.save(policy.state_dict(), args.out_pref+"_policy.pth")

    return policy, hist

def parse_args():
    parser = argparse.ArgumentParser(description='GRPO RL Mask Policy')

    parser.add_argument('--X_path', type=str, default='logits_ds_X (1).pkl',
                      help='path to X')
    parser.add_argument('--y_path', type=str, 
                      default='logits_ds_y (2).pkl',help='path to y')
    parser.add_argument('--G', type=int, default=10,
                      help='Num gens') 
    parser.add_argument('--epsilon', type=float, default=0.2,
                      help='clip limit')
    parser.add_argument('--beta', type=float, default=0.0001,
                      help='kl penalty weight')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='epochs')
    parser.add_argument('--update_ref_every', type=int, default=50,
                      help='epochs to update ref_policy every')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='policy hidden size')
    parser.add_argument('--n_layers', type=int, default=2,
                      help='policy layers')
    parser.add_argument('--n_heads', type=int, default=4,
                      help='policy heads')
    
    parser.add_argument('--out_pref', type=str, default="model",
                      help='Prefix for output files')
    
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
        
    policy, hist = train_loop(args)
    print(hist)




        
        
        

            
            
              


        

    


