import numpy as np
import tiktoken
from pathlib import Path

# Step 1: Locate the data directory
data_dir = Path(__file__).resolve().parent / "data"

# Step 2: Load the raw text
with open(data_dir / "input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
# Step 3: Train-validation split
n = int(len(raw_text) * 0.9)
train_text = raw_text[:n]
test_text = raw_text[n:]

# Step 4: Tokenization
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode(train_text)
test_ids = enc.encode(test_text)

# Step 5: Convert to Numpy arrays
train_ids = np.array(train_ids, dtype=np.uint16)
test_ids = np.array(test_ids, dtype=np.uint16)

# Step 6: Save the tokenized datasets
np.save(data_dir / "train.npy", train_ids)
np.save(data_dir / "test.npy", test_ids)
