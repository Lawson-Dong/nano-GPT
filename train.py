import torch
import numpy as np
from pathlib import Path
from config import GPTConfig
from contextlib import nullcontext
from model import nanoGPT
import time


config = GPTConfig()
block_size = config.block_size

data_dir = Path(__file__).resolve().parent / "data"
train_data = np.load(data_dir / "train.npy")
test_data = np.load(data_dir / "test.npy")

def get_batch(split):
    """
    from the dataset, sample a batch of inputs x and targets y
    """
    data = train_data if split == "train" else test_data
    
    # Sample random starting positions for the batch
    beginning_positions = torch.randint(
        len(data) - block_size, 
        (batch_size,)
    )
    
    x = torch.stack([torch.tensor(data[i:i+block_size].tolist(), dtype=torch.long) for i in beginning_positions])
    y = torch.stack([torch.tensor(data[i+1:i+1+block_size].tolist(), dtype=torch.long) for i in beginning_positions])
    # The position of y is the position of x shifts to right for 1 unit
    
    return x, y

# Training loop
# Define the training epochs

max_iters = 500
batch_size = 16
block_size = config.block_size
eval_interval = 100
eval_iters = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

model = nanoGPT(config).to(device)

# Define the datatype and context manager
# To enable mixed precision training
dtype = (
    "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
) if device == "cuda" else "float32"

ptdtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[dtype]

context_manager = (
    torch.amp.autocast(device_type=device, dtype=ptdtype)
    if device == "cuda" 
    else nullcontext()
)

# Define gradient scaler for mixed precision training
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device == "cuda")
)

# Define optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# Track the training loss and test loss during the training process every eval_interval steps
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ("train", "test"):
        total_loss = 0.0
        for _ in range(eval_iters):
            X, y = get_batch(split)
            X, y = X.to(device), y.to(device)
            with context_manager:
                _, loss = model(X, y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses

# Set up beginning time for measuring the time taken for each training step
t0 = time.time()

for step in range(max_iters):
    
    # Get a batch of training data
    # And move the data to the device (GPU or CPU)
    X_train, y_train = get_batch("train")
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Forward pass
    with context_manager:
        logits, loss = model(X_train, y_train)
        
    # Optimizer zero grad
    optimizer.zero_grad(set_to_none=True)
        
    # Backward pass
    scaler.scale(loss).backward()
    
    # Gradient clipping to prevent exploding gradients
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update parameters
    scaler.step(optimizer)
    scaler.update()


    if (step + 1) % eval_interval == 0 or step == max_iters - 1:
        dt = time.time() - t0
        tokens_per_second = batch_size * block_size * eval_interval / dt
        losses = estimate_loss()
        print(
            f"step {step + 1}/{max_iters}: "
            f"train loss {losses['train']:.4f}, "
            f"test loss {losses['test']:.4f}, "
            f"tokens per second: {tokens_per_second:.2f}"
        )
        t0 = time.time()
        
torch.save(model.state_dict(), data_dir / "model.pt")



