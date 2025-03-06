import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import datasets
import time
import plotext as plt
import copy

batch_size = 1024
available_gpus = [torch.device('cuda', i) for i in range(torch.cuda.device_count())]

print('start model init')
start_time = time.time()

model_name = 'SmerkyG/RWKV7-Goose-0.1B-World2.8-HF'
model_cpu = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
models = [copy.deepcopy(model_cpu).to(x).eval() for x in available_gpus]
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

for model in models:
    for param in model.parameters():
        param.requires_grad = False

print(f'finish model init, took {time.time() - start_time}')

print('start ds init')
start_time = time.time()

# todo shuffle
ds = datasets.load_dataset("JeanKaddour/minipile", split='train') # 1337497600 tokens
#ds = datasets.load_dataset('cerebras/SlimPajama-627B', streaming=True, split='train')
ds = ds.shuffle()

iterable_train_ds = iter(ds)
print(f'finish ds init, took {time.time() - start_time}')

class ResidualLoader():
    def __init__(self, dataset, model, tokenizer, batch_size):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # init state
        self.curr_state = model.forward(torch.zeros(batch_size,1, dtype = torch.long, device=self.model.device)).state
        self.slot_queue = [[] for _ in range(batch_size)]

        self.activations = {}
        self.handles = []
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.activations[name] = output
            return hook

        for module in self.model.named_modules():
            self.handles.append(module[1].register_forward_hook(getActivation(module[0])))

    def get_batch(self):
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
                    self.activations = {}
                    self.curr_state = self.model.forward(batch, state=self.curr_state).state
                    filtered_acts = {}
                    # extract residual acts after every time-mix and channel-mix module
                    for blk in range(12):
                        # post attn
                        filtered_acts[f'attn.{blk}'] = self.activations[f'model.blocks.{blk}'][0] - self.activations[f'model.blocks.{blk}.feed_forward'][0]
                        # post ffn
                        filtered_acts[f'ffn.{blk}'] = self.activations[f'model.blocks.{blk}'][0]
                    all_acts = [v.flatten(0,1) for v in filtered_acts.values()]

                    return torch.stack(all_acts, dim=1)



class TopKSAE(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            k=16,
            act_fn=nn.ReLU,
            device='cpu',
    ):
        super().__init__()
        self.encoder = nn.Linear(dim, hidden_features, device = device)
        self.act  = act_fn()
        self.decoder = nn.Linear(hidden_features, dim, device = device)
        self.num_active_features = k
        self.act_sum = torch.zeros(hidden_features, device = device).float()
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

class TopKMLPSAE(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            dense_hidden_features,
            k=16,
            act_fn=nn.ReLU,
            device='cpu',
    ):
        super().__init__()
        
        
        self.encoder_w1 = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(dense_hidden_features, dim, device=device)))
        self.encoder_b1 = nn.Parameter(data=torch.zeros(dense_hidden_features, device=device))
        self.encoder_act = nn.GELU()
        self.encoder_w2 = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(hidden_features, dense_hidden_features, device=device)))
        self.encoder_b2 = nn.Parameter(data=torch.zeros(hidden_features, device=device))
        
        self.act = act_fn()
        self.num_active_features = k

        self.decoder_w1 = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(dense_hidden_features, hidden_features, device=device)))
        self.decoder_b1 = nn.Parameter(data=torch.zeros(dense_hidden_features, device=device))
        self.decoder_act = nn.GELU()
        self.decoder_w2 = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(dim, dense_hidden_features, device=device)))
        self.decoder_b2 = nn.Parameter(data=torch.zeros(dim, device=device))
        
        self.act_sum = torch.zeros(hidden_features, device = device).float()
        self.act_ema = torch.zeros(hidden_features, device = device).float()
        self.decay = 0.96
        self.eps = 1e-8

    
    def forward(self, x):
        B, C = x.shape

        x = x - self.decoder_b2
        
        x = torch.einsum('bc,dc->bd', x, self.encoder_w1)
        x = x + self.encoder_b1
        x = self.encoder_act(x)
        x = torch.einsum('bd,nd->bn', x, self.encoder_w2)
        x = x + self.encoder_b2
        
        topk_values, topk_indices = torch.topk(x, self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).flatten(0).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act

        x = torch.einsum('bn,dn->bd', x, self.decoder_w1)
        x = x + self.decoder_b1
        x = self.decoder_act(x)
        x = torch.einsum('bd,cd->bc', x, self.decoder_w2)
        x = x + self.decoder_b2
        return x

class DenseTopKMLPSAE(nn.Module):
    def __init__(self, sae_list):
        super().__init__()
        self.encoder_w1 = nn.Parameter(torch.stack([sae.encoder_w1 for sae in sae_list]))
        self.encoder_b1 = nn.Parameter(torch.stack([sae.encoder_b1 for sae in sae_list]))
        self.encoder_act = sae_list[0].encoder_act
        self.encoder_w2 = nn.Parameter(torch.stack([sae.encoder_w2 for sae in sae_list]))
        self.encoder_b2 = nn.Parameter(torch.stack([sae.encoder_b2 for sae in sae_list]))
        
        self.act = sae_list[0].act
        self.num_active_features = sae_list[0].num_active_features
        
        self.decoder_w1 = nn.Parameter(torch.stack([sae.decoder_w1 for sae in sae_list]))
        self.decoder_b1 = nn.Parameter(torch.stack([sae.decoder_b1 for sae in sae_list]))
        self.decoder_act = sae_list[0].decoder_act
        self.decoder_w2 = nn.Parameter(torch.stack([sae.decoder_w2 for sae in sae_list]))
        self.decoder_b2 = nn.Parameter(torch.stack([sae.decoder_b2 for sae in sae_list]))
        
        self.register_buffer('act_sum', torch.stack([sae.act_sum for sae in sae_list]))
        self.register_buffer('act_ema', torch.stack([sae.act_ema for sae in sae_list]))
        
        self.decay = sae_list[0].decay
        self.eps = sae_list[0].eps
        
    
    def forward(self, x):
        # [B, R, C]
        x = x - self.decoder_b2
        
        x = torch.einsum('brc,rdc->brd', x, self.encoder_w1)
        x = x + self.encoder_b1
        x = self.encoder_act(x)
        x = torch.einsum('brd,rnd->brn', x, self.encoder_w2)
        x = x + self.encoder_b2
        
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

        x = torch.einsum('brn,rdn->brd', x, self.decoder_w1)
        x = x + self.decoder_b1
        x = self.decoder_act(x)
        x = torch.einsum('brd,rcd->brc', x, self.decoder_w2)
        x = x + self.decoder_b2
        return x


print('start loader init')
start_time = time.time()
loaders = [ResidualLoader(iterable_train_ds, model, tokenizer, batch_size) for model in models]
print(f'finish loader init, took {time.time() - start_time}')

print('start sae init')
start_time = time.time()
#saeList = [TopKSAE(768, 32768, k=8192, device = torch.device('cpu')) for i in range(24)]
#denseSaeList = [DenseTopKSAE(saeList[i:i + 3]).train().to(available_gpus[d]) for d, i in enumerate(range(0, 24, 3))]

baseSae = TopKMLPSAE(768, 32768, 4096, k=8192, device = torch.device('cpu'))
saeList = [copy.deepcopy(baseSae) for i in range(24)]
denseSaeList = [DenseTopKMLPSAE(saeList[i:i + 3]).train().to(available_gpus[d]) for d, i in enumerate(range(0, 24, 3))]

optimizers = [optim.AdamW(sae.parameters(), lr=3e-4, weight_decay=1e-4) for sae in denseSaeList]
scalers = [torch.amp.GradScaler() for sae in denseSaeList]
print(f'finish sae init, took {time.time() - start_time}')


# https://cdn.openai.com/papers/sparse-autoencoders.pdf
# via https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py 
def norm_MSE(pred, targ): return (((pred - targ) ** 2).mean(dim=-1) / (targ**2).mean(dim=-1)).mean()
criterion = norm_MSE

opt_steps = 0
steps_per_printout = 25
steps_per_histogram = 25
grad_accum_epochs = 64
curr_batch=0
eps = 1e-8
start_time = time.time()
have_more_data=True

print('starting train loop')
while(1):
    
    with torch.no_grad():
        targ_all_loaders = []
        for loader in loaders:
            with torch.autocast(device_type="cuda"):
                targ = loader.get_batch()
            if targ is None:
                have_more_data = False
                print("no more data")
                break
            # D * [B, L, C]
            targ_all_loaders += [targ.detach().float()]

    if not have_more_data: break
    for device_offset in range(len(available_gpus)):
        curr_batch += 1
        
        with torch.no_grad():

            # D * [B, R, C]
            targs = [targ_all_loaders[(d + device_offset) % len(available_gpus)][:, i:i + 3].to(available_gpus[d], non_blocking=True) for d, i in enumerate(range(0, 24, 3))]
            
        with torch.autocast(device_type="cuda",):
            preds = [sae(targ) for sae, targ in zip(denseSaeList, targs)]
            
        losses = [0 for _ in available_gpus]
        for i, (pred, targ) in enumerate(zip(preds, targs)):
            losses[i % len(available_gpus)] = losses[i % len(available_gpus)] + criterion(pred, targ) 
        losses = [loss / grad_accum_epochs for loss in losses]
        [scaler.scale(loss).backward() for loss, scaler in zip(losses, scalers)]
        
        # print k and do update
        do_print=True
        for sae in denseSaeList:
            if sae.num_active_features > 64:
                sae.num_active_features = sae.num_active_features - 1
                if do_print == True:
                    do_print=False
                    print(f"k = {denseSaeList[0].num_active_features}")
        
        if curr_batch % grad_accum_epochs == 0:
            opt_steps += 1
            [scaler.step(optimizer) for optimizer, scaler in zip(optimizers, scalers)]
            [scaler.update() for scaler in scalers]
            [optimizer.zero_grad(set_to_none=True) for optimizer in optimizers]
            
            # histograms
            if opt_steps % steps_per_histogram == 0:
                plt.hist(torch.stack([sae.act_sum.cpu() for sae in denseSaeList]).sum((0,1)).add(eps).log(), 50, label=f'total acts')
                plt.show()
                plt.clear_figure()
                plt.hist(torch.stack([sae.act_ema.cpu() for sae in denseSaeList]).sum((0,1)).add(eps).log(), 50, label='running acts')
                plt.show()
                plt.clear_figure()
                for sae in denseSaeList: sae.act_sum = sae.act_sum * 0
            
            if opt_steps % steps_per_printout == 0:
                # print training info
                print(f'tokens: {opt_steps * batch_size * grad_accum_epochs}, mse loss: {torch.tensor([loss.cpu() * grad_accum_epochs for loss in losses]).mean()}, avg step time: {(time.time() - start_time) / steps_per_printout}, tps: {(batch_size * grad_accum_epochs * steps_per_printout)/(time.time() - start_time)}')
                start_time = time.time()
              
    
denseSaeList = [x.to('cpu') for x in denseSaeList]

for idx, sae in enumerate(denseSaeList):
    for saeIdx, (enc_w, enc_b, dec_w, dec_b) in enumerate(
            zip(sae.encoder_w, sae.encoder_b, sae.decoder_w, sae.decoder_b)):
        state_dict = {}
        state_dict['encoder.weight'] = enc_w
        state_dict['encoder.bias'] = enc_b
        state_dict['decoder.weight'] = dec_w
        state_dict['decoder.bias'] = dec_b
        saeList[idx * 3 + saeIdx].load_state_dict(state_dict)

saeList = nn.ModuleList(saeList)

torch.save(saeList, "sae_checkpoint.pth")
