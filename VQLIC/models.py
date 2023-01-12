import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import ScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import LowerBound
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch.residual_vq import ResidualVQ, MultiLayerVQ, HierarchicalVQ

def build_net(mode, N, dim):
    if mode == 'ga':
        return nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            conv3x3(N, dim, stride=2),
            AttentionBlock(dim),
            ResidualBlock(dim, dim),
        )
    elif mode == 'gs':
        return nn.Sequential(
            AttentionBlock(dim),
            ResidualBlock(dim, dim),
            ResidualBlockUpsample(dim, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )
    elif mode == 'ha':
        return nn.Sequential(
            conv3x3(dim, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )
    elif mode == 'hs':
        return nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, dim * 2),            
        )
    elif mode =='hs_vw': # vector-wise # error
         return nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, quantizers * 2),
        )

class AutoEncoder(nn.Module):
    def __init__(self, N=128, dim=64):
        super().__init__()
        self.g_a = build_net('ga', N, dim)
        self.g_s = build_net('gs', N, dim)
        
    def forward(self, x):
        y = self.g_a(x)
        x_hat = self.g_s(y)
        return {
            'x_hat': x_hat,
        }
class FineTuningAE(nn.Module):
    def __init__(self, N=128, dim=64, quantizers=1, CB_size=512):
        super().__init__()
        self.AE = AutoEncoder(N, dim)
        self.quantizers = quantizers
        self.dim = dim
        if quantizers == 1:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = CB_size,    # codebook size
                decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 0.25,   # the weight on the commitment loss
                accept_image_fmap = True,
                kmeans_init = True,   # set to True
                kmeans_iters = 10,    # number of kmeans iterations to calculate the centroids for the codebook on init                
                threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2                
            )
        elif quantizers > 1:
            self.vq = MultiLayerVQ(
                num_quantizers = quantizers,
                dim = dim // quantizers,
                codebook_size = CB_size,    # codebook size
                decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 0.25,   # the weight on the commitment loss
                accept_image_fmap = True,
                kmeans_init = True,   # set to True
                kmeans_iters = 10,    # number of kmeans iterations to calculate the centroids for the codebook on init                
                threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2
            )
    def forward(self, x):
        y = self.AE.g_a(x)
        y_hat, y_id, commit, usage = self.vq(y)  # (b, Q, w, h), (b, Q, w, h), (b), (b)
        x_hat = self.AE.g_s(y_hat)
        return {
            "y":y,
            "y_hat":y_hat,
            "x_hat": x_hat,
            "usage": usage,
        }

class VQVAE(nn.Module):
    def __init__(self, N=128, dim=64, quantizers=1, CB_size=512):
        super().__init__()
        self.AE = AutoEncoder(N, dim)
        self.gaussian_conditional = GaussianConditional(None) # calculate likelihood
        self.lower_bound_l = LowerBound(1e-9)                 # likelihood lower bound
        self.quantizers = quantizers
        self.dim = dim
        if quantizers == 1:
            self.vq = VectorQuantize(
                dim = dim,
                codebook_size = CB_size,    # codebook size
                decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 0.25,   # the weight on the commitment loss
                accept_image_fmap = True,
                threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2                
            )
        elif quantizers > 1:
            self.vq = MultiLayerVQ(
                num_quantizers = quantizers,
                dim = dim // quantizers,
                codebook_size = CB_size,    # codebook size
                decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
                commitment_weight = 0.25,   # the weight on the commitment loss
                accept_image_fmap = True,
                threshold_ema_dead_code = 2  # should actively replace any codes that have an exponential moving average cluster size less than 2                
            )
            
    def load_AE(self, path):
        AE_params = torch.load(path)
        if 'AE' in AE_params:
            AE_params = AE_params['AE']
        self.AE.load_state_dict(AE_params)
            
    def calc_cross_entropy(self, symbol):
        ones = torch.ones_like(symbol).to(device).float()
        cross_entropy_from_N_01 = self.gaussian_conditional._likelihood(symbol, ones)
        cross_entropy_from_N_01 = self.lower_bound_l(cross_entropy_from_N_01)
        return cross_entropy_from_N_01

    def forward(self, x):
        y = self.AE.g_a(x)
        ce = self.calc_cross_entropy(y)
        z_likelihoods = torch.tensor(1)
        y_hat, y_id, commit, usage = self.vq(y)  # (b, Q, w, h), (b, Q, w, h), (b), (b)
        x_hat = self.AE.g_s(y_hat)
        return {
            "y":y,
            "y_hat":y_hat,            
            "x_hat": x_hat,
            "likelihoods": z_likelihoods,
            "commit": commit,
            "usage": usage,
            "cross": ce,
        }

class VQVAE_mixed_dims(VQVAE):
    def __init__(self, N=128, CB_size_list = [256, 256, 256, 256], dim_list = [8, 8, 16, 32], quantizers=4):
        super().__init__(N, sum(dim_list), quantizers, CB_size_list[0])
        self.dim = sum(dim_list)            
        self.dim_list = dim_list
        self.CB_size_list = CB_size_list
        # overriding self.vq
        self.vq = HierarchicalVQ(
            num_quantizers = quantizers,
            dim_list = dim_list,
            CB_size_list = CB_size_list,    # codebook size
            decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 0.25,   # the weight on the commitment loss
            accept_image_fmap = True,
        )

class Adapt_VQ(VQVAE):
    def __init__(self, N=128, dim=64, quantizers=1, CB_size=512):
        super().__init__(N, dim, quantizers, CB_size)
        self.entropy_bottleneck = EntropyBottleneck(channels = N)
        self.lower_bound_s = LowerBound(0.11)
        self.h_a = build_net('ha', N, dim)
        self.h_s = build_net('hs', N, dim)
        
    # ADD function to load pre-trained model
    def standardized(self, y, means, scales):
        standard_deviations = self.lower_bound_s(scales) # SD shouldn't be less than 0
        y_std = y - means
        y_std = y_std / standard_deviations
        ce_N01 = self.calc_cross_entropy(y_std)
        return y_std, ce_N01
            
    def destandardized(self, y_std, means, scales):
        standard_deviations = self.lower_bound_s(scales)        
        y_ = y_std * standard_deviations
        y_ = y_ + means
        return y_
    
    def represent_befor_quantize(self, y):
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_std, ce = self.standardized(y, means_hat, scales_hat)
        return y_std, ce, z_likelihoods, scales_hat, means_hat, z, z_hat
    def aux_loss(self):
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss    
    def forward(self, x):
        y = self.AE.g_a(x)
        y_std, ce, z_likelihoods, scales_hat, means_hat, z, z_hat = self.represent_befor_quantize(y)        
        y_std_hat, y_id, commit, usage = self.vq(y_std)  # (b, Q, w, h), (b, Q, w, h), (b), (b)
        y_hat = self.destandardized(y_std_hat, scales_hat, means_hat)
        x_hat = self.AE.g_s(y_hat)

        return {
            "y": y,
            "y_std":y_std,
            "y_std_hat": y_std_hat,
            "y_hat": y_std,
            "z":z,
            "z_hat": z_hat,
            "x_hat": x_hat,
            "likelihoods": z_likelihoods,            
            "commit": commit,
            "usage": usage,
            "cross": ce,
            "scales_hat":scales_hat,
            "means_hat":means_hat,
        }
    
class Adapt_VQ_mixed_dims(Adapt_VQ):
    def __init__(self, N=128, quantizers=4, CB_size_list = [256, 256, 256, 256], dim_list = [8, 8, 16, 32]):        
        super().__init__(N, sum(dim_list), quantizers, CB_size_list[0])
        self.dim = sum(dim_list)
        self.dim_list = dim_list
        self.CB_size_list = CB_size_list
        self.vq = HierarchicalVQ(
            num_quantizers = quantizers,
            dim_list = dim_list,
            CB_size_list = CB_size_list,    # codebook size
            decay = 0.99,               # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 0.25,   # the weight on the commitment loss
            accept_image_fmap = True,
        )
class Adapt_VQ_vectorwise(Adapt_VQ):
    def __init__(self, N=128, dim=64, quantizers=1, CB_size=512):
        super().__init__(N, dim, quantizers, CB_size)
        self.h_s = build_net('hs_vw', N, dim)
    def standardized(self, y, means, scales):
        standard_deviations = self.lower_bound_s(scales)        
        pd = self.dim // self.quantizers # dimensionality of the data processed in each iteration
        for i in range(self.quantizers):
            y_ = y[:, i*pd:(i+1)*pd] - means[:, i].unsqueeze(1)
            y_ = y_ / standard_deviations[:, i].unsqueeze(1)
            y_std = y_ if i == 0 else torch.cat([y_std, y_], 1)
        return y_std, ce_N01
    def destandardization(self, y, means, scales):
        standard_deviations = self.lower_bound_s(scales)        
        pd = self.dim // self.quantizers
        for i in range(self.quantizers):
            y_hat = y_std[:, i*pd:(i+1)*pd] * standard_deviations[:, i].unsqueeze(1)
            y_hat = y_hat + means[:, i].unsqueeze(1)
            y_ = y_hat if i == 0 else torch.cat([y_, y_hat], 1)            
        return y_

class Adapt_VQ_direct(Adapt_VQ):
    def __init__(self, N=128, dim=64, quantizers=1, CB_size=512):
        super().__init__(N, dim, quantizers, CB_size)
        self.h_s = build_net('hs_vw', N, dim)
    def direct_calc_gaussian(self, y):
        means = y.mean(1).unsqueeze(1)
        scales = ((y - means) ** 2).mean(1).unsqueeze(1)        
        SD = torch.sqrt(scales)
        
        gaussian_params = torch.cat([SD, means], 1)
        z = self.h_a(gaussian_params)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        gp_hat = self.h_s(z_hat)
        scales_hat, means_hat = gp_hat.chunk(2, 1)
        y_std, ce = self.standardized(y, means_hat, scales_hat)
        return y_std, ce, z_likelihoods, scales_hat, means_hat, gaussian_params, gp_hat
    
    def forward(self, x):
        y = self.AE.g_a(x)
        y_std, ce, z_likelihoods, scales_hat, means_hat, gp, gp_hat = self.direct_calc_gaussian(y)
        y_hat, id, commit, usage = self.vq(y_std)  # (b, Q, w, h), (b, Q, w, h), (b), (b)
        y_hat_ = self.destandardized(y_hat, scales_hat, means_hat) if hyper_enable else y_hat
        x_hat = self.AE.g_s(y_hat_)

        return {
            "x_hat": x_hat,
            "likelihoods": z_likelihoods,
            "commit": commit,
            "usage": usage,
            "cross": ce,
            "gp": gp,
            "gp_hat": gp_hat,            
        }
    
def get_model(model_name="VQVAE"):
    if model_name == "AutoEncoder":
        return AutoEncoder
    elif model_name == "VQVAE":
        return VQVAE
    elif model_name == "Adapt_VQ":
        return Adapt_VQ
    elif model_name == "FineTuningAE":
        return FineTuningAE
    else:
        print('search failed! return default: AE')
        return AutoEncoder