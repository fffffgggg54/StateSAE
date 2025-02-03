import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import fla
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import datasets
import time
import plotext as plt


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model_device = torch.device('cuda:0')

model_name = 'SmerkyG/RWKV7-Goose-0.1B-Pile-HF'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = model.to(model_device)
model = model.eval()
for param in model.parameters():
    param.requires_grad = False

# todo shuffle
ds = datasets.load_dataset("JeanKaddour/minipile")
iterable_train_ds = iter(ds['train'])

class StateLoader():
    def __init__(self, dataset, model, tokenizer, batch_size):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # init state
        self.curr_state = model.forward(torch.zeros(batch_size,1, dtype = torch.long, device=self.model.device)).state
        self.slot_queue = [[] for _ in range(batch_size)]

    def get_state_batch(self):
        with torch.no_grad():
            with torch.inference_mode():
                # reset state where needed
                for i, reset in enumerate([len(x) == 0 for x in self.slot_queue]):
                    if reset:
                        self.curr_state[0][:, i] *= 0
                        self.curr_state[1][:, i] *= 0
                        self.curr_state[2][:, i] *= 0
                try:
                    # get new sequence in slots with no remaining tokens left
                    self.slot_queue = [self.tokenizer(next(self.dataset)['text'], return_tensors="pt")['input_ids'][0] if len(x) == 0 else x for x in self.slot_queue]
                except:
                    # case no more inputs
                    return None
                # build batch out of first token from each slot
                batch = [x[0] for x in self.slot_queue]
                batch = torch.stack(batch).unsqueeze(-1).to(self.model.device)

                # remove first token from each slot
                self.slot_queue = [x[1:] for x in self.slot_queue]

                # forward model and get states
                self.curr_state = model.forward(batch, state=self.curr_state).state

                # V6: [Tensor(B, C, L), Tensor(B, n_h, d_h, d_h, L), Tensor(B, C, L)]
                # V7: [Tensor(L, B, C), Tensor(L, B, n_h, d_h, d_h), Tensor(L, B, C)]

                return self.curr_state[1]


class TopKRoutingBiasedSAE(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            k=16,
            lr=3e-4,
            act_fn=nn.ReLU,
            device='cpu',
    ):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_features, device = device)
        self.act  = act_fn()
        self.decoder = nn.Linear(hidden_features, dim, device = device)
        self.register_buffer('feature_bias', torch.zeros(hidden_features, device = device).float(), persistent=True)
        self.lr=lr
        self.num_active_features = k
        self.register_buffer('act_sum', torch.zeros(hidden_features, device = device).float())
        self.register_buffer('act_ema', torch.zeros(hidden_features, device = device).float())
        self.decay = 0.96
        self.eps = 1e-8
    
    def forward(self, x):
        # [B, C]
        x = x - self.decoder.bias
        x = self.encoder(x)
        
        topk_values, topk_indices = torch.topk(x + self.feature_bias.detach(), self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act
                feature_error = self.act_ema.mean().add(self.eps).log() - self.act_ema.add(self.eps).log()
                self.feature_bias = (1-self.lr) * self.feature_bias + self.lr * feature_error * feature_error.abs() > 5

        x = self.decoder(x)
        return x
'''
class TopKRoutingBiasedSAEWithPerStateLoRA(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            num_layers = 12,
            num_heads = 12,
            k=16,
            r=128,
            lr=3e-4,
            act_fn=nn.ReLU,
    ):
        super().__init__()
        self.bias_per_state = nn.Parameter(data=torch.randn(1, num_layers, num_heads, dim) * 0.1)

        self.encoder = nn.Linear(dim, hidden_features)

        self.encoder_lora_down = nn.Parameter(data=torch.randn(dim, r) * 0.02)
        self.encoder_lora_per_state = nn.Parameter(data=torch.randn(num_layers, num_heads, r, r) * 0.02)
        self.encoder_lora_up = nn.Parameter(data=torch.randn(r, hidden_features) * 0.02)

        self.act  = act_fn()

        self.decoder = nn.Linear(hidden_features, dim, bias=False)

        self.decoder_lora_down = nn.Parameter(data=torch.randn(hidden_features, r) * 0.02)
        self.decoder_lora_per_state = nn.Parameter(data=torch.randn(num_layers, num_heads, r, r) * 0.02)
        self.decoder_lora_up = nn.Parameter(data=torch.randn(r, dim) * 0.02)

        self.register_buffer('feature_bias', torch.zeros(hidden_features).float(), persistent=True)
        self.lr=lr
        self.num_active_features = k
        self.register_buffer('act_sum', torch.zeros(hidden_features).float())
        self.register_buffer('act_ema', torch.zeros(hidden_features).float())
        self.decay = 0.96
        self.eps = 1e-8
    
    def forward(self, x):
        B, L, n_h, C = x.shape

        x = x - self.bias_per_state
        x1 = self.encoder(x)

        x2 = x @ self.encoder_lora_down
        x2 = x2.unsqueeze(-2) @ self.encoder_lora_per_state.unsqueeze(0)
        x2 = x2 @ self.encoder_lora_up
        x = x1 + x2.squeeze(-2)

        x = x.flatten(0, 2)
        
        topk_values, topk_indices = torch.topk(x + self.feature_bias.detach(), self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act
                feature_error = self.act_ema.mean().add(self.eps).log() - self.act_ema.add(self.eps).log()
                self.feature_bias = self.feature_bias + self.lr * feature_error
        
        x = x.reshape(B, L, n_h, -1)

        x1 = self.decoder(x)

        x2 = x @ self.decoder_lora_down
        x2 = x2.unsqueeze(-2) @ self.decoder_lora_per_state.unsqueeze(0) 
        x2 = x2 @ self.decoder_lora_up
        x = x1 + x2.squeeze(-2)
        x = x + self.bias_per_state
        return x
'''
'''
class TopKRoutingBiasedSAEWithFullPerStateLoRA(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            num_layers = 12,
            num_heads = 12,
            k=16,
            r=128,
            act_fn=nn.ReLU,
    ):
        super().__init__()
        self.bias_per_state = nn.Parameter(data=torch.randn(1, num_layers, num_heads, dim) * 0.1)

        self.encoder = nn.Linear(dim, hidden_features)

        self.encoder_lora_down = nn.Parameter(data=torch.randn(1, num_layers, num_heads, dim, r) * 0.02)
        self.encoder_lora_up = nn.Parameter(data=torch.randn(1, num_layers, num_heads, r, hidden_features) * 0.02)

        self.act  = act_fn()

        self.decoder = nn.Linear(hidden_features, dim, bias=False)

        self.decoder_lora_down = nn.Parameter(data=torch.randn(1, num_layers, num_heads, hidden_features, r) * 0.02)
        self.decoder_lora_up = nn.Parameter(data=torch.randn(1, num_layers, num_heads, r, dim) * 0.02)

        self.register_buffer('feature_bias', torch.zeros(hidden_features).float(), persistent=True)
        self.lr=1e-3
        self.num_active_features = k
        self.register_buffer('act_sum', torch.zeros(hidden_features).float())
    
    def forward(self, x):
        B, L, n_h, C = x.shape

        x = x - self.bias_per_state
        x1 = self.encoder(x)

        x2 = x.unsqueeze(-2) @ self.encoder_lora_down
        x2 = x2 @ self.encoder_lora_up
        x = x1 + x2.squeeze(-2)

        x = x.flatten(0, 2)
        
        topk_values, topk_indices = torch.topk(x + self.feature_bias.detach(), self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        
        with torch.no_grad():
            feature_act = (mask > 0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                feature_error = feature_act.mean() - feature_act
                self.feature_bias = self.feature_bias + self.lr * feature_error
        
        x = x.reshape(B, L, n_h, -1)

        x1 = self.decoder(x)

        x2 = x.unsqueeze(-2) @ self.decoder_lora_down
        x2 = x2 @ self.decoder_lora_up
        x = x1 + x2.squeeze(-2)
        x = x + self.bias_per_state
        return x
'''
'''
class RWKVStateSAE(nn.Module):
    def __init__(self, sample_state, hidden_features):
        super().__init__()
        L, B, n_h, d_h, _ = sample_state.shape
        self.sae_list = nn.ModuleList([
                nn.ModuleList([
                    SingleRWKVStateSAE(sample_state, hidden_features) for _ in range(n_h)])
                for _ in range(L)])
        self.last_l1 = None

    def forward(self, x):
        out = torch.zeros_like(x)
        #self.last_l1 = torch.zeros(1).to(x)
        x = x.transpose(1,2)
        for l, layer in enumerate(x):
            for h, head in enumerate(layer):
                out[l, :, h] = self.sae_list[l][h](head)
                #curr_l1 = self.sae_list[l][h].last_l1
                #self.last_l1 = self.last_l1 + curr_l1
        return out
'''
batch_size = 32


available_gpus = [torch.device('cuda', i) for i in range(torch.cuda.device_count())]

state_loader = StateLoader(iterable_train_ds, model, tokenizer, batch_size)
#sae = RWKVStateSAE(state_loader.curr_state[1], 1024).to('cuda:0')
#sae = TopKRoutingBiasedSAEWithFullPerStateLoRA(4096, 4096 * 16, num_layers = 12, num_heads = 12, k=4096*2, r=32).to(sae_device)
#sae = TopKRoutingBiasedSAEWithPerStateLoRA(4096, 4096 * 16, num_layers = 12, num_heads = 12, k=256, r=512, lr=1e-4).to(sae_device)
saeList = [TopKRoutingBiasedSAE(4096, 4096*4, k=128, lr=1e-4, device = available_gpus[i % len(available_gpus)]) for i in range(12*12)]
#saeList = [x.to(available_gpus[i % len(available_gpus)]) for i, x in enumerate(saeList)]
saeList = [sae.train() for sae in saeList]
optimizers = [optim.AdamW(sae.parameters(), lr=1e-4, weight_decay=3e-3) for sae in saeList]
#criterion = nn.MSELoss()

# https://cdn.openai.com/papers/sparse-autoencoders.pdf
# via https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py 
def norm_MSE(pred, targ): return (((pred - targ) ** 2).mean(dim=-1) / (targ**2).mean(dim=-1)).mean()
criterion = norm_MSE


steps_per_printout = 25
steps_per_histogram = 250
i=0
eps = 1e-8
start_time = time.time()
while(1):
    i += 1
    
    with torch.autocast(device_type="cuda"):
        with torch.no_grad():
            # cuda:0
            state = state_loader.get_state_batch()
            if state is None:
                break
            state = state.detach()
            states = state.transpose(1, 2).flatten(-2).flatten(0,1) # [L * n_h, B, d_h**2]
            stateList = [x.to(available_gpus[i % len(available_gpus)], non_blocking=True) for i, x in enumerate(states)]
        pred_states = [sae(state) for sae, state in zip(saeList, stateList)]
        losses = [0 for _ in len(available_gpus)]
        for i, (pred, targ) in enumerate(zip(pred_states, states)):
            losses[i] = losses[i] + criterion(pred, targ) 


        
    [loss.backward() for loss in losses]
    [optimizer.step() for optimizer in optimizers]
    [optimizer.zero_grad(set_to_none=True) for optimizer in optimizers]
    
    '''
    if sae.num_active_features > 128:
        sae.num_active_features = sae.num_active_features - 1
    '''
    if i % steps_per_printout == 0:
        print(f'tokens: {i * batch_size}, mse loss: {torch.tensor([loss.cpu() for loss in losses]).mean()}, avg step time: {(time.time() - start_time) / steps_per_printout}')
        start_time = time.time()
    if i % steps_per_histogram == 0:
        for sae in saeList: plt.hist(sae.act_sum.cpu().add(eps).log(), 50, label=f'sae {i} acts')
        plt.show()
        plt.clear_figure()
        for sae in saeList: plt.hist(sae.act_ema.cpu().add(eps).log(), 50, label='running acts')
        plt.show()
        plt.clear_figure()
        for sae in saeList: sae.act_sum = sae.act_sum * 0










    


