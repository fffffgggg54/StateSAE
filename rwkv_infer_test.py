import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import time
from torch.utils.data import DataLoader

model_name = 'SmerkyG/RWKV7-Goose-0.1B-Pile-HF'
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = model.eval()

#ds = datasets.load_dataset("JeanKaddour/minipile")
ds = datasets.load_dataset('cerebras/SlimPajama-627B', streaming=True, split='train').shuffle()

iterable_train_ds = iter(ds)
device = torch.device('cuda:0')
#device = torch.device('cpu')
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

batch_size = 256


class StateLoader():
    def __init__(self, dataset, model, tokenizer, batch_size):
        self.dataset = dataset
        self.loader = iter(DataLoader(dataset, num_workers=8, prefetch_factor=8, batch_size=1))
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # init state
        self.curr_state = model.forward(torch.zeros(batch_size,1, dtype = torch.long, device=device)).state
        self.slot_queue = [[] for _ in range(batch_size)]

    def get_state_batch(self):
        with torch.no_grad():
            # reset state where needed
            for i, reset in enumerate([len(x) == 0 for x in self.slot_queue]):
                if reset:
                    self.curr_state[0][:, i] *= 0
                    self.curr_state[1][:, i] *= 0
                    self.curr_state[2][:, i] *= 0

            # get new sequence in slots with no remaining tokens left
            self.slot_queue = [self.tokenizer(next(self.loader)['text'], return_tensors="pt")['input_ids'][0] if len(x) == 0 else x for x in self.slot_queue]

            # build batch out of first token from each slot
            batch = [x[0] for x in self.slot_queue]
            batch = torch.stack(batch).unsqueeze(-1).to(device)

            # remove first token from each slot
            self.slot_queue = [x[1:] for x in self.slot_queue]

            # forward model and get states
            self.curr_state = model.forward(batch, state=self.curr_state).state

            # V6: [Tensor(B, C, L), Tensor(B, n_h, d_h, d_h, L), Tensor(B, C, L)]
            # V7: [Tensor(L, B, C), Tensor(L, B, n_h, d_h, d_h), Tensor(L, B, C)]

        return self.curr_state[1]


#state_loader = StateLoader(iterable_train_ds, model, tokenizer, batch_size)
state_loader = StateLoader(ds, model, tokenizer, batch_size)

while(1):
    start_time = time.time()
    with torch.inference_mode():
        with torch.no_grad():
            state = state_loader.get_state_batch().detach()
    print(f'tok/sec: {batch_size/(time.time() - start_time)}')
