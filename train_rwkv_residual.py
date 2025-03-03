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

batch_size = 2048

# 1 gpu, gh200

model_name = 'SmerkyG/RWKV7-Goose-0.1B-World2.8-HF'
model_cpu = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = copy.deepcopy(model_cpu).to('cuda:0').eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


for param in model.parameters():
    param.requires_grad = False

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


class TopKSAEWithPerActBias(nn.Module):
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
        self.decoder_b = nn.Parameter(data=nn.init.kaiming_uniform_(torch.empty(num_layers * 2, dim, device=device)))
        
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
        



state_loader = StateLoader(iterable_train_ds, model, tokenizer, batch_size)
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

optimizers = optim.AdamW(sae.parameters(), lr=1e-4, weight_decay=1e-4)


# https://cdn.openai.com/papers/sparse-autoencoders.pdf
# via https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/loss.py 
def norm_MSE(pred, targ): return (((pred - targ) ** 2).mean(dim=-1) / (targ**2).mean(dim=-1)).mean()
criterion = norm_MSE

opt_steps = 0
steps_per_printout = 1000
steps_per_histogram = 1000
grad_accum_epochs = 1
curr_batch=0
eps = 1e-8
start_time = time.time()
have_more_data=True
while(1):
    
    with torch.no_grad():
        state_all_loaders = []
        for state_loader in state_loaders:
            with torch.autocast(device_type="cuda"):
                state = state_loader.get_state_batch()
            if state is None:
                have_more_data = False
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
    if not have_more_data: break
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
        
        # print k and do update
        do_print=True
        for sae in denseSaeList:
            if sae.num_active_features > 16:
                sae.num_active_features = sae.num_active_features - 1
                if do_print == True:
                    do_print=False
                    print(f"k = {denseSaeList[0].num_active_features}")
        
        if curr_batch % grad_accum_epochs == 0:
            opt_steps += 1
            [optimizer.step() for optimizer in optimizers]
            [optimizer.zero_grad(set_to_none=True) for optimizer in optimizers]
            #[scheduler.step() for scheduler in schedulers]
            
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
                print(f'tokens: {opt_steps * batch_size * grad_accum_epochs}, mse loss: {torch.tensor([loss.cpu() for loss in losses]).mean()}, avg step time: {(time.time() - start_time) / steps_per_printout}, tps: {(batch_size * grad_accum_epochs * steps_per_printout)/(time.time() - start_time)}')
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
    








    


