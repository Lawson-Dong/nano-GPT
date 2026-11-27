# NanoGPT is a decoder-only transformer model, which is a simplified version of the GPT architecture

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # Embedding dimension must be divisible by number of heads
        
        # Use one Linear layer to get Q, K, V simultaneously
        # This is for all attention head
        # W_Q, W_K and W_V are trainable parameters, containing in this Linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # Output Projection: Project the Output after attention to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Key attribute
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Set the causal mask buffer
        # This is a lower triangular matrix, which is used to prevent attention to future tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size))
        )
        
    def forward(self, x):
        # B: Batch size, T: Sequence length, C: Embedding dimension
        # The input x has shape (B, T, C)
        B, T, C = x.size()
        
        # The first step is to get Q, K and V 
        # The shape of self.c_attn(x) is (B, T, 3 * C)
        # We then split it into Q, K and V, each of shape (B, T, C)
        # d_k, d_v, d_q are all equal to C, the embedding dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # To achieve multi-head attention, we need to reshape Q, K and V to (B, nh, T, hs)
        # Head size (hs) is defined as n_embd // n_head
        # Pay attention to transpose the second and third dimensions, because we want to have the head dimension before the sequence length dimension
        # since we want to compute the attention for each head separately, and PyTorch's batch matrix multiplication is designed to operate on the last two dimensions of the input tensors
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute the attention scores
        # To conduct matrix multiplication, we are going to use @
        # Tensor product: (B, nh, T_Q, hs) @ (B, nh, hs, T_K) -> (B, nh, T_Q, T_K)
        att = ( q @ k.transpose(-2, -1) ) * ( 1.0 / math.sqrt(k.size(-1)) )
        
        # Apply the causal mask to prevent attention to future tokens
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Softmax to get the attention weights
        # This is a normalization step, which ensures that the attention weights sum to 1
        # The softmax function is applied along the last dimension, which corresponds to the sequence length of the keys (T_K)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Compute the weighted sum of the values
        # Tensor product: (B, nh, T_Q, T_K) @ (B, nh, T_K, hs) -> (B, nh, T_Q, hs)
        y = att @ v
        
        # Reshape the output back to (B, T, C)
        # We need to merge the multi-head outputs back into a single tensor
        # contiguous() is used to ensure that the tensor is stored in a contiguous chunk of memory, which is required for the view operation
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply the output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.mlp = nn.Sequential(
            # The first linear layer projects the input tensor to a higher dimensional space
            # 256 -> 4 * 256 = 1024
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.mlp(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            CausalSelfAttention(config),
        )
        self.block_2 = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            MLP(config),
        )
        
    def forward(self, x):
        # Residual connections: Add the input to the output of each sub-layer
        x = x + self.block_1(x)
        x = x + self.block_2(x)
        return x
    
class nanoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),                   # wte: Tensor shape change: (B, T) -> (B, T, C) 
                wpe = nn.Embedding(config.block_size, config.n_embd),                   # wpe: Tensor shape change: (T,) -> (T, C)
                drop = nn.Dropout(config.dropout),
                h = nn.Sequential(*[Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        
        # Shared weights with wte (word embedding) and the language modeling head
        # Tensor shape change: (B, T, C) -> (B, T, vocab_size)
        # vocab_size = 50304, which is the size of the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # lm_head is the transpose of wte
        # nn.Linear help us do the transpose
        # The shape of the weight tensor is (vocab_size, C) for both lm_head and wte 
        
        # Weights initialization
        self.apply(self._init_weights)
        
        # The scaling of the residuals
        # This is a technique used to stabilize the training of deep neural networks
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, 
                    mean=0.0, 
                    std=0.02/math.sqrt(2 * config.n_layer)
                )
        
    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def forward(self, idx, targets=None):
        '''
        Args:
            idx: (B, T) tensor of integers, where B is the batch size and T is the sequence length
            targets: (B, T) tensor of integers, where B is the batch size and T is the sequence length
        Returns:
            logits: (B, T, vocab_size) tensor of floats, where B is the batch size, T is the sequence length and vocab_size is the size of the vocabulary
        '''
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # The first step: embedding
        # wte: ( B, T ) -> (B, T, C)
        # wpe: uses arange to create positional embeddings [0, 1, 2, ..., T-1] -> (T, C)
        # broadcasting: (B, T, C) + (T, C) -> (B, T, C)
        token_embd = self.transformer.wte(idx)
        pos_embd = self.transformer.wpe(torch.arange(t, device=idx.device))
        x = self.transformer.drop(token_embd + pos_embd)
        
        # Pass the input through the transformer blocks
        # Multi-layer Transformer
        for block in self.transformer.h:
            x = block(x)
            
        # Final layer normalization
        x = self.transformer.ln_f(x)
        
        # Language modeling head and return logits
        logits = self.lm_head(x)
        
        # Calculate the loss when training 
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),       # Predicted logits predicted by the model
                targets.view(-1),                       # True labels (targets)
                ignore_index=-1                         # If the target is -1, it will be ignored in the loss calculation
            )
            
        return logits, loss
        
        
            
            
        