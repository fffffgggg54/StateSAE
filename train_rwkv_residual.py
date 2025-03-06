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

batch_size = 1536

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 1 gpu, gh200

print('start model init')
start_time = time.time()

model_name = 'SmerkyG/RWKV7-Goose-0.1B-World2.8-HF'
model_cpu = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = copy.deepcopy(model_cpu).to('cuda:0').eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)




for param in model.parameters():
    param.requires_grad = False
print(f'finish model init, took {time.time() - start_time}')
print('start ds init')
start_time = time.time()

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
            num_layers = 12,
            k=16,
            act_fn=nn.ReLU,
            device='cpu',
    ):
        super().__init__()
        self.decoder_b = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(1, num_layers * 2, dim, device=device)))
        
        self.encoder_w = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(hidden_features, dim, device=device)))
        self.encoder_b = nn.Parameter(data=torch.zeros(hidden_features, device=device))
        
        self.act = act_fn()
        self.num_active_features = k

        
        self.decoder_w = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(dim, hidden_features, device=device)))
        
        self.act_sum = torch.zeros(hidden_features, device = device).float()
        self.act_ema = torch.zeros(hidden_features, device = device).float()
        self.decay = 0.96
        self.eps = 1e-8

    
    def forward(self, x):
        B, L, C = x.shape

        x = x - self.decoder_b
        
        x = torch.einsum('blc,dc->bld', x, self.encoder_w)
        x = x + self.encoder_b
        
        topk_values, topk_indices = torch.topk(x, self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).flatten(0, 1).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act

        x = torch.einsum('bld,cd->blc', x, self.decoder_w)
        x = x + self.decoder_b
        return x
        
class TopKMLPSAE(nn.Module):
    def __init__(
            self,
            dim,
            hidden_features,
            dense_hidden_features,
            num_layers = 12,
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
        self.decoder_b2 = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(1, num_layers * 2, dim, device=device)))
        
        self.act_sum = torch.zeros(hidden_features, device = device).float()
        self.act_ema = torch.zeros(hidden_features, device = device).float()
        self.decay = 0.96
        self.eps = 1e-8

    
    def forward(self, x):
        B, L, C = x.shape

        x = x - self.decoder_b2
        
        x = torch.einsum('blc,dc->bld', x, self.encoder_w1)
        x = x + self.encoder_b1
        x = self.encoder_act(x)
        x = torch.einsum('bld,nd->bln', x, self.encoder_w2)
        x = x + self.encoder_b2
        
        topk_values, topk_indices = torch.topk(x, self.num_active_features, dim=-1, sorted=False)

        mask = torch.zeros_like(x).scatter_(-1, topk_indices, 1)
        x = x * mask
        
        x = self.act(x)

        # desire log-normal distribution
        with torch.no_grad():
            feature_act = (mask > 0).flatten(0, 1).sum(dim=0).float()
            self.act_sum += feature_act
            if self.training:
                self.act_ema = self.decay * self.act_ema + (1-self.decay) * feature_act

        x = torch.einsum('bln,dn->bld', x, self.decoder_w1)
        x = x + self.decoder_b1
        x = self.decoder_act(x)
        x = torch.einsum('bld,cd->blc', x, self.decoder_w2)
        x = x + self.decoder_b2
        return x

print('start loader init')
start_time = time.time()
loader = ResidualLoader(iterable_train_ds, model, tokenizer, batch_size)
print(f'finish loader init, took {time.time() - start_time}')

print('start sae init')
start_time = time.time()
#sae = TopKSAE(768, 2**16, k=2**12, device='cuda:0')
sae = TopKMLPSAE(768, 2**16, 4096, k=2**12, device='cuda:0')

optimizer = optim.AdamW(sae.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.amp.GradScaler()

print(f'finish sae init, took {time.time() - start_time}')


# https://cdn.openai.com/papers/sparse-autoencoders.pdf
# via https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py 
def norm_MSE(pred, targ): return (((pred - targ) ** 2).mean(dim=-1) / (targ**2).mean(dim=-1)).mean()
criterion = norm_MSE

opt_steps = 0
steps_per_printout = 25
steps_per_histogram = 25
do_plot = False
grad_accum_epochs = 16
curr_batch=0
eps = 1e-5
start_time = time.time()
have_more_data=True
print('starting train loop')
while(1):
    
    with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                residual_batch = loader.get_batch()
            if residual_batch is None:
                have_more_data = False
                print("no more data")
                break
    if not have_more_data: break
    curr_batch += 1
    with torch.autocast(device_type="cuda", dtype = torch.bfloat16):
        preds = sae(residual_batch)

    loss = criterion(preds, residual_batch) / grad_accum_epochs
    scaler.scale(loss).backward()

    if curr_batch % grad_accum_epochs == 0:
        opt_steps += 1
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if sae.num_active_features > 32:
            sae.num_active_features = sae.num_active_features - 1
            print(f"k = {sae.num_active_features}")
            
        # histograms
        if opt_steps % steps_per_histogram == 0 and do_plot:
            plt.hist(sae.act_sum.cpu().add(eps).log(), 50, label=f'total acts')
            plt.show()
            plt.clear_figure()
            plt.hist(sae.act_ema.cpu().add(eps).log(), 50, label='running acts')
            plt.show()
            plt.clear_figure()
            sae.act_sum = sae.act_sum * 0
        
        if opt_steps % steps_per_printout == 0:
            # print training info
            print(f'tokens: {opt_steps * batch_size * grad_accum_epochs}, mse loss: {loss.cpu()}, avg step time: {(time.time() - start_time) / steps_per_printout}, tps: {(batch_size * grad_accum_epochs * steps_per_printout)/(time.time() - start_time)}')
            start_time = time.time()

            
            
    
sae = sae.cpu()

torch.save(sae, "rwkv7-0.1B-world2.8-resMLPSAE-minipile.pth")
    








    


