import torch

if torch.cuda.is_available():
    print('Using GPU')
    DEVICE = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    DEVICE = 'cpu'


XMEM_CONFIG = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 256,
    'min_mid_term_frames': 7,
    'max_mid_term_frames': 20,
    'max_long_term_elements': 10000,
}
