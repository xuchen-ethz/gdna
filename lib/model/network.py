""" The code is based on https://github.com/lioryariv/idr with adaption. """

import torch
import numpy as np
import torch.nn as nn

from lib.model.helpers import mask_dict, expand_cond, grid_sample_feat

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            d_in,
            d_out,
            width,
            depth,
            geometric_init=True,
            bias=1.0,
            skip_in=-1,
            weight_norm=True,
            multires=0,
            pose_cond_layer=-1,
            pose_cond_dim=-1,
            pose_embed_dim=-1,
            shape_cond_layer=-1,
            shape_cond_dim=-1,
            shape_embed_dim=-1,
            latent_cond_layer=-1,
            latent_cond_dim=-1,
            latent_embed_dim=-1,
            feat_cond_dim=0,
            feat_cond_layer=[],
            **kwargs
    ):
        super().__init__()

        dims = [d_in] + [width]*depth + [d_out]
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.cond_names = []

        self.pose_cond_layer = pose_cond_layer
        self.pose_cond_dim = pose_cond_dim
        self.pose_embed_dim = pose_embed_dim
        if len(pose_cond_layer)>0:
            self.cond_names.append('pose')
        if pose_embed_dim > 0:
            self.lin_p0 = nn.Linear(pose_cond_dim, pose_embed_dim)
            self.pose_cond_dim = pose_embed_dim

        self.shape_cond_layer = shape_cond_layer
        self.shape_cond_dim = shape_cond_dim
        self.shape_embed_dim = shape_embed_dim
        if len(shape_cond_layer)>0:
            self.cond_names.append('betas')
        if shape_embed_dim > 0:
            self.lin_p1 = nn.Linear(shape_cond_dim, shape_embed_dim)
            self.shape_cond_dim = shape_embed_dim

        self.latent_cond_layer = latent_cond_layer
        self.latent_cond_dim = latent_cond_dim
        self.latent_embed_dim = latent_embed_dim
        if len(latent_cond_layer)>0:
            self.cond_names.append('latent')
        if latent_embed_dim > 0:
            self.lin_p2 = nn.Linear(latent_cond_dim, latent_embed_dim)
            self.latent_cond_dim = latent_embed_dim

        self.feat_cond_layer = feat_cond_layer
        self.feat_cond_dim = feat_cond_dim
        
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 == self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            input_dim = dims[l]
            if l in self.pose_cond_layer:
                input_dim += self.pose_cond_dim
            if l in self.shape_cond_layer:
                input_dim += self.shape_cond_dim
            if l in self.latent_cond_layer:
                input_dim += self.latent_cond_dim
            if l in self.feat_cond_layer:
                input_dim += self.feat_cond_dim

            lin = nn.Linear(input_dim, out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(input_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if l in self.latent_cond_layer:
                        torch.nn.init.normal_(lin.weight[:, -latent_cond_dim:], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l == self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(input_ch - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond, input_feat=None,  mask=None, return_feat=False, spatial_feat=False, val_pad=0, normalize=False):
        
        input_dim = input.ndim

        if normalize:
            input = input.clone()
            input[..., 1] += 0.28 
            input[...,-1] *= 4

        if input_dim == 3:
            n_batch, n_point, n_dim = input.shape
            if mask is None:
                mask = torch.ones( (n_batch, n_point), device=input.device, dtype=torch.bool)

            if spatial_feat:
                cond = { key:grid_sample_feat(cond[key], input) for key in cond if key in self.cond_names}
            else:
                cond = { key:expand_cond(cond[key], input) for key in cond if key in self.cond_names}

            cond = mask_dict(cond, mask)

            input = input[mask]

            if len(self.feat_cond_layer) > 0:
                input_feat = input_feat[mask]

        if len(self.pose_cond_layer) > 0:
            input_pose_cond = cond['thetas']
            if self.pose_embed_dim>0:
                input_pose_cond = self.lin_p0(input_pose_cond)

        if len(self.shape_cond_layer) > 0:
            input_shape_cond = cond['betas']
            if self.shape_embed_dim>0:
                input_shape_cond = self.lin_p1(input_shape_cond)      

        if len(self.latent_cond_layer) > 0:
            input_latent_cond = cond['latent']
            if self.latent_embed_dim>0:
                input_shape_cond = self.lin_p2(input_latent_cond)     
        
        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.pose_cond_layer:
                x = torch.cat([x, input_pose_cond], dim=-1)

            if l in self.shape_cond_layer:
                x = torch.cat([x, input_shape_cond], dim=-1)

            if l in self.latent_cond_layer:
                x = torch.cat([x, input_latent_cond], dim=-1)

            if l in self.feat_cond_layer:
                x = torch.cat([x, input_feat], dim=-1)

            if l == self.skip_in:
                x = torch.cat([x, input_embed], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
                feat = x.clone()

        if input_dim == 3:
            x_full = torch.ones( (n_batch, n_point, x.shape[-1]), device=x.device) * val_pad
            x_full[mask] = x
            x = x_full

            if return_feat:
                feat_full = torch.ones( (n_batch, n_point, feat.shape[-1]), device=x.device) * val_pad
                feat_full[mask] = feat
                feat = feat_full
                
        if return_feat:
            return x, feat
        else:
            return x
        


""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim