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
import copy

import pytorch_optimizer

batch_size = 2048
available_gpus = [torch.device('cuda', i) for i in range(torch.cuda.device_count())]

model_name = 'SmerkyG/RWKV7-Goose-0.1B-Pile-HF'
model_cpu = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
models = [copy.deepcopy(model_cpu).to(x).eval() for x in available_gpus]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

for model in models:
    for param in model.parameters():
        param.requires_grad = False

# todo shuffle
ds = datasets.load_dataset("JeanKaddour/minipile", split='train') # 1337497600 tokens
#ds = datasets.load_dataset('cerebras/SlimPajama-627B', streaming=True, split='train')
ds = ds.shuffle()

iterable_train_ds = iter(ds)

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
                with torch.autocast(device_type="cuda"):
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
                    batch = torch.stack(batch).unsqueeze(-1).to(self.model.device, non_blocking=True)

                    # remove first token from each slot
                    self.slot_queue = [x[1:] for x in self.slot_queue]

                    # forward model and get states
                    self.curr_state = self.model.forward(batch, state=self.curr_state).state

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
        #self.register_buffer('feature_bias', torch.zeros(hidden_features, device = device).float(), persistent=True)
        self.lr=lr
        self.num_active_features = k
        #self.register_buffer('act_sum', torch.zeros(hidden_features, device = device).float())
        self.act_sum = torch.zeros(hidden_features, device = device).float()
        #self.register_buffer('act_ema', torch.zeros(hidden_features, device = device).float())
        self.act_ema = torch.zeros(hidden_features, device = device).float()
        self.decay = 0.96
        self.eps = 1e-8
    
    def forward(self, x):
        # [B, C]
        x = x - self.decoder.bias
        x = self.encoder(x)
        
        #topk_values, topk_indices = torch.topk(x + self.feature_bias.detach(), self.num_active_features, dim=-1, sorted=False)

        topk_values, topk_indices = torch.topk(x, self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act
                #feature_error = self.act_ema.mean().add(self.eps).log() - self.act_ema.add(self.eps).log()
                #self.feature_bias = (1-self.lr) * self.feature_bias + self.lr * feature_error * feature_error.abs() > 6

        x = self.decoder(x)
        return x
        
        
class DenseTopKSAE(nn.Module):
    ''''
    def __init__(
            self,
            dim,
            hidden_features,
            replicas,
            k=16,
            act_fn=nn.ReLU,
            device='cpu',
    ):
        super().__init__()
        #self.encoder = nn.Linear(dim, hidden_features, device = device)

        self.encoder_w = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(hidden_features, dim, device=device)).repeat(replicas, 1, 1))
        self.encoder_b = nn.Parameter(data=torch.zeros(replicas, hidden_features, device=device))
        self.act = act_fn()
        
        
        self.decoder_w = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(dim, hidden_features, device=device)).repeat(replicas, 1, 1))
        self.decoder_b = nn.Parameter(data=torch.zeros(replicas, dim, device=device))
        self.num_active_features = k
        self.register_buffer('act_sum', torch.zeros(replicas, hidden_features, device = device).float())
        self.register_buffer('act_ema', torch.zeros(replicas, hidden_features, device = device).float())
        self.decay = 0.96
        self.eps = 1e-8
    
    '''
    def __init__(self, sae_list):
        super().__init__()
        self.encoder_w = nn.Parameter(torch.stack([sae.encoder.weight for sae in sae_list]))
        self.encoder_b = nn.Parameter(torch.stack([sae.encoder.bias for sae in sae_list]))
        
        self.act = sae_list[0].act
        self.num_active_features = sae_list[0].num_active_features
        
        self.decoder_w = nn.Parameter(torch.stack([sae.decoder.weight for sae in sae_list]))
        self.decoder_b = nn.Parameter(torch.stack([sae.decoder.bias for sae in sae_list]))
        
        self.register_buffer('act_sum', torch.stack([sae.act_sum for sae in sae_list]))
        self.register_buffer('act_ema', torch.stack([sae.act_ema for sae in sae_list]))
        
        self.decay = sae_list[0].decay
        self.eps = sae_list[0].eps
        
    
    def forward(self, x):
        # [B, R, C]
        x = x - self.decoder_b
        #x = x @ self.encoder_w
        x = torch.einsum('brc,rdc->brd', x, self.encoder_w)
        x = x + self.encoder_b
        
        topk_values, topk_indices = torch.topk(x, self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act

        x = torch.einsum('brd,rcd->brc', x, self.decoder_w)
        x = x + self.decoder_b
        return x.squeeze(-1)
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


state_loaders = [StateLoader(iterable_train_ds, model, tokenizer, batch_size) for model in models]
#sae = RWKVStateSAE(state_loader.curr_state[1], 1024).to('cuda:0')
#sae = TopKRoutingBiasedSAEWithFullPerStateLoRA(4096, 4096 * 16, num_layers = 12, num_heads = 12, k=4096*2, r=32).to(sae_device)
#sae = TopKRoutingBiasedSAEWithPerStateLoRA(4096, 4096 * 16, num_layers = 12, num_heads = 12, k=256, r=512, lr=1e-4).to(sae_device)
#saeList = [TopKRoutingBiasedSAE(4096, 4096*4, k=128, lr=1e-4, device = available_gpus[i % len(available_gpus)]) for i in range(2*12)]
#saeList = [TopKRoutingBiasedSAE(64, 64*128, k=16, lr=1e-4, device = available_gpus[i % len(available_gpus)]) for i in range(12*12)]
saeList = [TopKRoutingBiasedSAE(64, 64*128, k=64*64, lr=1e-4, device = torch.device('cpu')) for i in range(12*12)]

#saeList = [x.to(available_gpus[i % len(available_gpus)]) for i, x in enumerate(saeList)]
#saeList = [sae.train() for sae in saeList]

#optimizers = [optim.AdamW(sae.parameters(), lr=1e-4, weight_decay=1e-4) for sae in saeList]
#schedulers = [optim.lr_scheduler.CyclicLR(optimizer, step_size_down=5000, base_lr=1e-5, max_lr=1e-3) for optimizer in optimizers]
#criterion = nn.MSELoss()

denseSaeList = [DenseTopKSAE(saeList[i:i + 18]).train().to(available_gpus[d]) for d, i in enumerate(range(0, 144, 18))]
#denseSaeList = [DenseTopKSAE(64, 64*128, 18, k=64*64, device=gpu).to(gpu) for gpu in available_gpus]
#optimizers = [optim.AdamW(sae.parameters(), lr=1e-4, weight_decay=1e-4) for sae in denseSaeList]
optimizers = [pytorch_optimizer.SignSGD(sae.parameters(), lr=1e-3, weight_decay=1e-4) for sae in denseSaeList]


# https://cdn.openai.com/papers/sparse-autoencoders.pdf
# via https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py 
def norm_MSE(pred, targ): return (((pred - targ) ** 2).mean(dim=-1) / (targ**2).mean(dim=-1)).mean()
criterion = norm_MSE

opt_steps = 0
steps_per_printout = 1
steps_per_histogram = 1
grad_accum_epochs = 256
curr_batch=0
eps = 1e-8
start_time = time.time()
while(1):
    
    with torch.no_grad():
        state_all_loaders = []
        for state_loader in state_loaders:
            with torch.autocast(device_type="cuda"):
                state = state_loader.get_state_batch()
            if state is None:
                print("no more data")
                break
            #state_all_loaders += [state.detach()[:, :, :2].transpose(1, 2).flatten(-2).flatten(0,1)]
            # [L * n_h, B, d_h]
            state_all_loaders += [(state.detach()[:, :, :12].transpose(1, 2).flatten(0,1).float() @ torch.ones(64, 1, device = state.device)).flatten(-2)]
    '''
    for state in state_all_loaders:
        curr_batch += 1
        
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                if state is None:
                    break
                state = state.detach() # [L, B, n_h, d_h, d_h]
                state = state[:, :, :2]
                states = state.transpose(1, 2).flatten(-2).flatten(0,1) # [L * n_h, B, d_h**2]
                states = [x.to(available_gpus[i % len(available_gpus)], non_blocking=True) for i, x in enumerate(states)]

    '''
    for device_offset in range(len(available_gpus)):
        curr_batch += 1
        
        with torch.no_grad():
            '''
            states = [
                state_all_loaders[(sae_id + device_offset) % len(available_gpus)][sae_id].to(available_gpus[sae_id % len(available_gpus)], non_blocking=True)
                for sae_id in range(len(saeList))
            ]
            '''
            # D * [R, B, d_h]
            states = [state_all_loaders[(d + device_offset) % len(available_gpus)][i:i + 18].to(available_gpus[d], non_blocking=True) for d, i in enumerate(range(0, 144, 18))]
            states = [state.transpose(0, 1) for state in states]

            #states = [x.to(available_gpus[i % len(available_gpus)], non_blocking=True) for i, x in enumerate(states)]
            
        
        pred_states = [sae(state) for sae, state in zip(denseSaeList, states)]
        losses = [0 for _ in available_gpus]
        for i, (pred, targ) in enumerate(zip(pred_states, states)):
            losses[i % len(available_gpus)] = losses[i % len(available_gpus)] + criterion(pred, targ) 

        #print(states[0].mean())
        #print(pred_states[0].mean())
        #print(states[0].shape)
            
        [loss.backward() for loss in losses]
        if curr_batch % grad_accum_epochs == 0:
            opt_steps += 1
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad(set_to_none=True) for optimizer in optimizers]
            #[scheduler.step() for scheduler in schedulers]
            do_print=True
            for sae in denseSaeList:
                if sae.num_active_features > 16:
                    sae.num_active_features = sae.num_active_features - 1
                    if do_print == True:
                        do_print=False
                        print(f"k = {denseSaeList[0].num_active_features}")
                
        
        if opt_steps % steps_per_printout == 0:
            print(f'tokens: {opt_steps * batch_size * grad_accum_epochs}, mse loss: {torch.tensor([loss.cpu() for loss in losses]).mean()}, avg step time: {(time.time() - start_time) / steps_per_printout}, tps: {(batch_size * grad_accum_epochs)/((time.time() - start_time) / steps_per_printout)}')
            start_time = time.time()
        '''
        if curr_batch % steps_per_histogram == 0:
            plt.hist(torch.stack([sae.act_sum.cpu() for sae in saeList]).sum(0).add(eps).log(), 50, label=f'total acts')
            plt.show()
            plt.clear_figure()
            plt.hist(torch.stack([sae.act_ema.cpu() for sae in saeList]).sum(0).add(eps).log(), 50, label='running acts')
            plt.show()
            plt.clear_figure()
            for sae in saeList: sae.act_sum = sae.act_sum * 0
        '''
        if curr_batch % steps_per_histogram == 0:
            plt.hist(torch.stack([sae.act_sum.cpu() for sae in denseSaeList]).sum((0,1)).add(eps).log(), 50, label=f'total acts')
            plt.show()
            plt.clear_figure()
            plt.hist(torch.stack([sae.act_ema.cpu() for sae in denseSaeList]).sum((0,1)).add(eps).log(), 50, label='running acts')
            plt.show()
            plt.clear_figure()
            for sae in denseSaeList: sae.act_sum = sae.act_sum * 0
            
            
    
denseSaeList = [x.to('cpu') for x in denseSaeList]

for idx, sae in enumerate(denseSaeList):
    for saeIdx, (enc_w, enc_b, dec_w, dec_b) in enumerate(
            zip(sae.encoder_w, sae.encoder_b, sae.decoder_w, sae.decoder_b)):
        state_dict = {}
        state_dict['encoder.weight'] = enc_w
        state_dict['encoder.bias'] = enc_b
        state_dict['decoder.weight'] = dec_w
        state_dict['decoder.bias'] = dec_b
        saeList[idx * 18 + saeIdx].load_state_dict(state_dict)

saeList = nn.ModuleList(saeList)

torch.save(saeList, "sae_checkpoint.pth")
    








    


