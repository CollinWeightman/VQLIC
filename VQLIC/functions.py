import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import shutil
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import os

class AE_loss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
           
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)  if ("likelihoods" in output) else torch.tensor(0)        
#         out["bpp_loss"] = torch.tensor(0)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"]
        return out
class FTAVQ_loss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["psnr"] = compute_psnr(output["x_hat"], target)        
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)  if ("likelihoods" in output) else torch.tensor(0)        
        out["y_mse"] = F.mse_loss(output["y_hat"], output["y"]) if ("y_hat" in output) else torch.tensor(0)
        out["loss"] = self.lmbda * 255**2 *  out["mse_loss"] + out["bpp_loss"]
        return out

class E2E_AVQ_loss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["psnr"] = compute_psnr(output["x_hat"], target)        
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)  if ("likelihoods" in output) else torch.tensor(0)        
        out["y_mse"] = F.mse_loss(output["y_hat"], output["y"]) if ("y_hat" in output) else torch.tensor(0)
        out["loss"] = self.lmbda * 255**2 *  out["mse_loss"] + out["bpp_loss"] + output["commit"].sum()
        return out
    
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def configure_optimizers_old(net):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    # ga and gs
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and not n.startswith("h")
    }
    # EntropyBottleneck in CompressAI liberay
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # ha and hs
    hyper_parameters = {
        n
        for n, p in net.named_parameters()
        if n.startswith("h") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters & hyper_parameters
    union_params = parameters | aux_parameters | hyper_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4,)
    out = []
    out.append(optimizer)
    if len(hyper_parameters) == 0 and len(aux_parameters) == 0:
        return out
    
    hyper_optimizer = optim.Adam((params_dict[n] for n in sorted(hyper_parameters)), lr=1e-4,)
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=1e-3,)
    out.append(hyper_optimizer)
    out.append(aux_optimizer)
    return out

def configure_optimizers(net):
    # ga, gs, (ha, hs, EB.bias, EB.matrix, EB.factor)
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    # EntropyBottleneck in CompressAI liberay (EB.quantiles)
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4,)
    out = []
    out.append(optimizer)
    if len(aux_parameters) == 0:
        return out
    
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=1e-3,)
    out.append(aux_optimizer)
    return out

def configure_optimizers_separate(net):
    # ha and hs and EB.bias, EB.matrix, EB.factor
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.startswith("AE") and not n.endswith(".quantiles") and p.requires_grad
    }
    # EntropyBottleneck in CompressAI liberay (EB.quantiles)
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4,)
    out = []
    out.append(optimizer)
    if len(aux_parameters) == 0:
        return out
    
    aux_optimizer = optim.Adam((params_dict[n] for n in sorted(aux_parameters)), lr=1e-3,)
    out.append(aux_optimizer)
    return out
def train_one_epoch(
    epoch, model,train_dataloader, optimizer_list, criterion, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        for j in range(len(optimizer_list)):
            optimizer_list[j].zero_grad()        
        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer_list[0].step()        
        if len(optimizer_list) > 1:
            aux_loss = model.aux_loss()
            aux_loss.backward()
            optimizer_list[1].step()

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_score = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr_score.update(compute_psnr(out_net["x_hat"], d))
            if hasattr(model, 'aux_loss'):
                aux_loss.update(model.aux_loss())            
    log_s = f'{loss.avg.item()}, {mse_loss.avg}, {bpp_loss.avg}, {aux_loss.avg}, {psnr_score.avg}\n'
    with open('log_training.csv', 'a') as f:
        f.write(log_s)
        
    print(log_s[:-1])
    return loss.avg

def save_checkpoint(state, is_best, filename):
    torch.save(state, f"{filename}.pth.tar")
    if is_best:
        shutil.copyfile(f"{filename}.pth.tar", f"{filename}_best.pth.tar")

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_mse(a, b):
    mse = torch.mean((a - b)**2).item()
    return mse

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    retval = torch.log(out_net['likelihoods']).sum() / (-math.log(2) * num_pixels) if 'likelihoods' in out_net else 0
    return retval

def eval_images(net, dir_path, version, data_ch_bpp):
    psnr = AverageMeter()
    msssim = AverageMeter()
    bitrate = AverageMeter()
    
    files = [dir_path + f for f in os.listdir(dir_path)]
    for i in range(len(files)):
        retval = eval_image(net, files[i], version, data_ch_bpp)
        psnr.update(retval[0])
        msssim.update(retval[1])
        bitrate.update(retval[2])
    
    return psnr.avg, msssim.avg, bitrate.avg

def save_model(net, version, CBs):
    if not os.path.exists(f'{version}.tar'):
        torch.save({}, f'{version}.tar')
    models = torch.load( f'{version}.tar')
    state = {
        "AE" : net.AE.state_dict(),
        f"{CBs}": net.state_dict(),
    }
    models.update(state)
    torch.save(models, f'{version}.tar')

def eval_image(net, img_path, version, data_ch_bpp):
    device = next(net.parameters()).device
    img = Image.open(img_path).convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device = device)
    with torch.no_grad():
        out_net = net.forward(x)
    out_net['x_hat'].clamp_(0, 1)
    psnr = compute_psnr(x, out_net["x_hat"])
    msssim = compute_msssim(x, out_net["x_hat"])
    bitrate = compute_bpp(out_net) + data_ch_bpp
    log_s = f"{version}_{int(data_ch_bpp*8*8)}, {os.path.basename(img_path)}, {psnr}, {msssim}, {bitrate}\n"
    with open('pic_log.csv', 'a') as f:
        f.write(log_s)
    return psnr, msssim, bitrate
        
def get_mu_and_sigma(symbol):
    symbol_flat = symbol.view(-1)    
    mean = symbol_flat.mean()
    mean_flat = torch.zeros_like(symbol_flat) + mean    
    sigma = ((symbol_flat - mean_flat) ** 2).mean().sqrt()
    return mean, sigma

def log_net_summery(
    args,
    now_epochs,
    net,
    version,
    data_ch_bpp,
    time_min,
    out_net,
):
    codebook_bit = int(data_ch_bpp * 8 * 8)    
    retval = eval_images(net, args.dataset + '/test/', version, data_ch_bpp)
    retval2 = eval_images(net, args.dataset + '/test2/', version, data_ch_bpp)            
    (psnr, msssim, total_bpp) = [(retval[i]*18 + retval2[i]*6)/24 for i in range(len(retval))]
    
    log_s = f"version, {version}_{codebook_bit}\n"
    log_s = log_s + f"mse wt, {args.lmbda}\n"
    
    usage = out_net['usage']
    y = out_net['y']
    y_hat = out_net['y_hat']
    
    m, s = get_mu_and_sigma(y)
    log_s = log_s + f"ymu, {m}\n"
    log_s = log_s + f"ysigma, {s}\n"
    
    if 'y_std' in out_net:
        y_std = out_net['y_std']
        m, s = get_mu_and_sigma(y_std)
        log_s = log_s + f"y_mu, {m}\n"
        log_s = log_s + f"y_sigma, {s}\n"    
    
    log_s = log_s + f"PSNR, {psnr}\n"
    log_s = log_s + f"ms-ssim, {msssim}\n"
    log_s = log_s + f"BPP, {total_bpp}\n"
    log_s = log_s + f"time(m), {time_min}\n"
    log_s = log_s + f"epochs, {now_epochs}\n"
    log_s = log_s + f"sec per ep., {time_min / now_epochs}\n"

    
    usage_list = []
    mse_list = []
    dim_list = []
    dim_num = args.vector_dim // args.quantizers
    for i in range(args.quantizers):
        dim_list.append(dim_num)
        
    if not usage.shape == torch.Size([]):
        for i in range(args.quantizers):    
            start = 0 if i == 0 else sum(dim_list[:i])
            end = dim_list[0] if i == 0 else sum(dim_list[:i+1])
            mse = F.mse_loss(y[:, start:end], y_hat[:, start:end])    
            mse_list.append(mse.item())
            usage_list.append(usage[i].item())
    else:
        mse = F.mse_loss(y, y_hat)
        mse_list.append(mse.item())
        usage_list.append(usage.item())

    log_s = log_s + f"usage"
    for i in range(args.quantizers):
        log_s = (log_s + f",{usage_list[i]}") 
    log_s = log_s + f"\n part_mse"
    for i in range(args.quantizers):
        log_s = (log_s + f",{mse_list[i]}" )        
    log_s = log_s + '\n\n'
    with open('finish_record.csv','a') as f:        
        f.write(log_s)