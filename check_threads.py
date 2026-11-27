import torch, os
print(os.cpu_count())
print(torch.get_num_threads())
print(torch.__config__.parallel_info()) 