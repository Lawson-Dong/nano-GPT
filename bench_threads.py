import time
import torch
from config import GPTConfig
from model import nanoGPT

config = GPTConfig()
model = nanoGPT(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 16
# Use random data 
# The meaning of two following lines is: generate a random tensor with shape (16, 128) using the random numbers between 0 and 50303
x = torch.randint(config.vocab_size, (batch_size, config.block_size))
y = torch.randint(config.vocab_size, (batch_size, config.block_size))

def train_step():
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

for n in [2, 4, 6, 8, 12]:
    torch.set_num_threads(n)
    for _ in range(5):         # Warm up
        train_step()
    t0 = time.time()
    for _ in range(100):
        train_step()
    ms = (time.time() - t0) / 100 * 1000
    tok_s = batch_size * config.block_size / (ms / 1000)
    print(f"{n:2d} Threads: {ms:6.0f} ms/step, {tok_s:7.0f} tok/s")