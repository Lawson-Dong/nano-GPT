from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False

    
    

