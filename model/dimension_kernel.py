import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0,include_input = True,input_dim=3):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : include_input,
                'input_dims' : input_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# Positional encoding
class Trigonometric_kernel:
    def __init__(self, L = 10, input_dim = 3, include_input=True):

        self.L = L
 
        self.embed_fn, self.out_ch= get_embedder(L,include_input = include_input, input_dim=input_dim)

    '''
    INPUT
     x: input vectors (N,C) 

     OUTPUT

     pos_kernel: (N, calc_dim(C) )
    '''
    def __call__(self, x):
        return self.embed_fn(x)

    def calc_dim(self, dims=0): #? dims的意义是啥
        return self.out_ch

# class Gaussian_Kernel:

#     def __init__(self, L, input_dim, scale=1):

#         self.scale = scale
#         self.input_dim =  input_dim
#         self.L = L

#         B = torch.normal(0, scale, size = (input_dim, L))

#         self.embed_fn = lambda x, eo = B : torch.cat([torch.sin((2*math.pi*x) @ B), torch.cos((2*math.pi*x) @ B)], dim = -1)

#     def __call__(self, x):
#         return self.embed_fn(x)

#     def calc_dim(self, dims=0):
#         return 2

class Gaussian_Kernel(nn.Module): 

    def __init__(self, dim_in, dim_embed, ffm_scale=16., trainable=False):
        super(Gaussian_Kernel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_embed * 2

        B = torch.normal(0., ffm_scale, size=(dim_embed, dim_in))
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B) 
    def calc_dim(self, dims=0):
        return self.dim_out

    def __call__(self, x):
        # print(x.shape)
        # print(self.B.shape)
        y = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)




