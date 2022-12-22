from functools import partial
import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

class ModResidualVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []

        for layer in self.layers:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))
        return quantized_out, all_indices, all_losses

class MultiLayerVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
    def forward(self, x):
        cb_dim = self.layers[0].codebook_dim
        for i in range(self.num_quantizers):
            part_x = x[:, i*cb_dim:(i + 1)*cb_dim, :, :]
            quantized, indices, loss, usage = self.layers[i](part_x)   # (b, Q, w, h), (b, w, h), (b)
            quantized_cat = quantized if i == 0 else torch.cat([quantized_cat, quantized], 1)
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)
#         perplexity = perplexity_cat.mean()
        return quantized_cat, indices_cat, loss_cat, perplexity_cat # (b, Q, w, h), (b, Q, w, h), (b), (b)

    def vq_recon(self, indices_cat):
        for i in range(self.num_quantizers):
            recon = self.layers[i].codebook[indices_cat[:, i]]  # (b, w, h, Q)
            recon = recon.permute(0, 3, 1, 2)                   # (b, Q, w, h)
            recon_cat = recon if i == 0 else torch.cat([recon_cat, recon], 1)
        return recon_cat

    
    
class HierarchicalVQ(nn.Module):
    def __init__(
        self,
        *,
        num_quantizers,
        dim_list,
        CB_size_list,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(dim_list[i], CB_size_list[i], **kwargs) for i in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.dim_list = dim_list
        self.CB_size_list = CB_size_list

    def forward(self, x):
        cb_dim = self.layers[0].codebook_dim
        for i in range(self.num_quantizers):
            start = 0 if i == 0 else sum(self.dim_list[:i])
            end = self.dim_list[0] if i == 0 else sum(self.dim_list[:i+1])            
            part_x = x[:, start:end, :, :]            
            quantized, indices, loss, usage = self.layers[i](part_x)   # (b, Q, w, h), (b, w, h), (b)
            quantized_cat = quantized if i == 0 else torch.cat([quantized_cat, quantized], 1)
            indices_cat = indices.unsqueeze(1) if i == 0 else torch.cat([indices_cat, indices.unsqueeze(1)], 1)
            loss_cat = loss if i == 0 else torch.cat([loss_cat, loss], 0)
            usage = usage.unsqueeze(0)
            perplexity_cat = usage if i == 0 else torch.cat([perplexity_cat, usage], 0)
#         perplexity = perplexity_cat.mean()
        return quantized_cat, indices_cat, loss_cat, perplexity_cat # (b, Q, w, h), (b, Q, w, h), (b), (b)

    