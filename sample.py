import torch
import torch.nn.functional as F

@torch.inference_mode()

def sample(model, x, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate new tokens from the model given a context x
    """
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
        
        logits, _ = model(x_cond)
        
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v,_ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, idx_next), dim=1)
        
    return x