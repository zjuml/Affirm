import argparse
import datetime
import os
# import sys
# sys.path.append("/home/huhuajin/TSLANet/Forecasting/Adaptive_Frequency_Filters")

import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from data_factory import data_provider
from utils import save_copy_of_files, random_masking_3D, str2bool
from Adaptive_Frequency_Filters.affnet.layers import ConvLayer, get_normalization_layer, get_activation_fn
from mamba_ssm import Mamba
from low_denoise import LowFrequencyDenoiser
import torch.nn.functional as F

class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x
    

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim,opts, hidden_size,num_blocks=8,hidden_size_factor=1,norm_layer=nn.LayerNorm):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        # self.norm1 = norm_layer(dim)
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        # self.act = nn.ReLU()

        
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.act = self.build_act_layer(opts=opts)
        self.act2 = self.build_act_layer(opts=opts)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask

    def forward(self, x_in):
        print(f"x_in:{x_in}")
        print(f"x_in.shape:{x_in.shape}")
        B, N, C = x_in.shape
        B1, C1, H, W = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
         
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        x_fft_1 = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        origin_ffted = x_fft_1 
        # x_norm = self.norm1(x_weighted.real)
        # # x_norm = x_norm.unsqueeze(1)
        # # x_norm = x_norm.transpose(1, 2)
        # x_relu = self.act(x_norm)
        # x_norm_twice = self.norm1(x_relu)
        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', origin_ffted.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', origin_ffted.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', origin_ffted.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', origin_ffted.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )
        x_fft_linear = torch.stack([o2_real, o2_imag], dim=-1)
        x_fft_linear = F.softshrink(x_fft_linear, lambd=self.sparsity_threshold)
        x_fft_linear = torch.view_as_complex(x_fft_linear)
        x_fft_linear = x_fft_linear.reshape(B1, C1, x.shape[3], x.shape[4])

        x_fft_linear = x_fft_linear * origin_ffted
        x_fft_linear = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        # x_fft_linear = x_fft_linear.type(dtype)
         


        if args.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask_high = self.create_adaptive_high_freq_mask(x_fft)
            a = LowFrequencyDenoiser(threshold_param=0.5)
            # print(f"a:{a}")
            freq_mask_low = a.create_adaptive_low_freq_mask(x_fft=x_fft)
            x_masked = x_fft * freq_mask_high.to(x.device)
            x1 = x_masked.clone()
            x_masked_low = x1 * freq_mask_low.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)

            # weight_low = torch.view_as_complex(self.complex_weight_low)

            x_weighted_high = x_masked * weight_high 

            x_weighted_low = x_masked_low * weight_high
            
            x_weighted_high_result = x_weighted_high * x_fft_linear
            x_weighted_origin = x_weighted * x_fft_linear
            x_weighted_low_result = x_weighted_low * x_fft_linear
            x_weighted = x_weighted_low_result + x_weighted_high_result + x_weighted_origin

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        # x_weighted = x_weighted  + x_fft_linear
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., d_model=128, bias=True, drop_path=0., norm_layer=nn.LayerNorm, configs1=0):
        super().__init__()
        self.configs = configs1
        self.d_model = d_model
        self.d_inner = 128
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim,opts=configs1, hidden_size=self.d_inner,num_blocks=8,hidden_size_factor=1,norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.act = nn.SiLU()
        self.conv3d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            # bias=conv_bias,
            kernel_size=self.configs.dconv1,
            # groups=self.d_inner,
            padding=self.configs.dconv1 - 1,
            # **factory_kwargs,
        )
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.configs.ch_ind==1:
            self.d_model_param1=1
            # self.d_model_param2=14

        else:
            self.d_model_param1=self.configs.n1
            # self.d_model_param2=self.configs.n1

        # self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        # self.act = nn.SiLU()
        self.mamba1=Mamba(dim,d_state=self.configs.d_state,d_conv=self.configs.dconv1,expand=self.configs.e_fact)
        print(f"self.mamba1:{self.mamba1}")

        self.mamba2=Mamba(dim,d_state=self.configs.d_state,d_conv=self.configs.dconv2,expand=self.configs.e_fact)
        print(f"self.mamba2:{self.mamba2}")

        # self.mamba3=Mamba(d_model=self.d_model_param1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        # self.mamba4=Mamba(d_model=self.configs.n1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)

    def forward(self, x):
        # Check if both ASB and ICB are true
        # if args.Mamba and args.ASB:
        # print(f"x_2:{x}")
        # print(f"x.shape:{x.shape}")
        if args.Mamba and args.ASB:
            a = self.norm1(x)
            # print(f"a:{a}")
            # print(f"a.shape:{a.shape}")
            b = self.asb(a)
            # print(f"b:{b}")
            # print(f"b.shape:{b.shape}")
            c = self.norm2(b)
            # print(f"c:{c}")
            # print(f"c.shape:{c.shape}")
            d = self.mamba1(c)
            # print(f"d:{d}")
            # print(f"d.shape:L{d.shape}")
            x1 = self.norm1(d)
            # print(f"x1:{x1}")
            # print(f"x1.shape:{x1.shape}")
            x1_1 = self.act(x1)
            # print(f"x1_1:{x1_1}")
            # print(f"x1_1.shape:{x1_1.shape}")
            x1_2 = self.drop_path(x1_1)
            # print(f"x1_2:{x1_2}")
            # print(f"x1_2.shape:{x1_2.shape}")

            

        

            e = self.mamba2(c)
            # print(f"e:{e}")
            # print(f"e.shape:{e.shape}")
            x2 = self.norm1(e)
            x2_1 = self.act(x2)
            x2_2 = self.drop_path(x2_1)
            # print(f"x2_2:{x2_2}")
            # print(f"x2_2.shape:{x2_2.shape}")
            
            d = self.in_proj(c)
            # print(f"d:{d}")
            # print(f"d.shape:{d.shape}")
            # print(f"x1:{x1}")
            # print(f"x1.shape:{x1.shape}")
            # print(f"x2_2:{x2_2}")
            # print(f"x2_2.shape:{x2_2.shape}")

            out1 = x1 * x2_2 * d
            # print(f"out1:{out1}")
            # print(f"out1.shape:{out1.shape}")
            out2 = x2 * x1_2 * d
            # print(f"out2:{out2}")
            # print(f"out2.shape:{out2.shape}")
           
    
            out3 = out1 + out2
            out3 = out3.transpose(1, 2)
            # f = self.conv3d(out1 + out2)
            f = self.conv3d(out3)
            f = f.transpose(1,2)
            g = self.norm1(f)
            # g = g.transpose(1,2)
            # print(f"f:{f}")
            # print(f"f.shape:{f.shape}")
            # print(f"x.shape:{x.shape}")
            # g = g.transpose(1,2)
            # print(f"f:{f}")
            # print(f"f.shape:{f.shape}")
            x = x + g
            # x = x + self.drop_path(self.mamba1(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is trbaue
        elif args.Mamba:
            a = self.norm1(x)
            # print(f"a:{a}")
            # print(f"a.shape:{a.shape}")
            b = self.asb(a)
            # print(f"b:{b}")
            # print(f"b.shape:{b.shape}")
            c = self.norm2(b)
            # print(f"c:{c}")
            # print(f"c.shape:{c.shape}")
            d = self.mamba1(c)
            # print(f"d:{d}")
            # print(f"d.shape:L{d.shape}")
            x1 = self.norm1(d)
            # print(f"x1:{x1}")
            # print(f"x1.shape:{x1.shape}")
            x1_1 = self.act(x1)
            # print(f"x1_1:{x1_1}")
            # print(f"x1_1.shape:{x1_1.shape}")
            x1_2 = self.drop_path(x1_1)
            # print(f"x1_2:{x1_2}")
            # print(f"x1_2.shape:{x1_2.shape}")

            

        

            e = self.mamba2(c)
            # print(f"e:{e}")
            # print(f"e.shape:{e.shape}")
            x2 = self.norm1(e)
            x2_1 = self.act(x2)
            x2_2 = self.drop_path(x2_1)
            # print(f"x2_2:{x2_2}")
            # print(f"x2_2.shape:{x2_2.shape}")
            
            d = self.in_proj(c)
            # print(f"d:{d}")
            # print(f"d.shape:{d.shape}")
            # print(f"x1:{x1}")
            # print(f"x1.shape:{x1.shape}")
            # print(f"x2_2:{x2_2}")
            # print(f"x2_2.shape:{x2_2.shape}")

            out1 = x1 * x2_2 * d
            # print(f"out1:{out1}")
            # print(f"out1.shape:{out1.shape}")
            out2 = x2 * x1_2 * d
            # print(f"out2:{out2}")
            # print(f"out2.shape:{out2.shape}")
           
    
            out3 = out1 + out2
            out3 = out3.transpose(1, 2)
            # f = self.conv3d(out1 + out2)
            f = self.conv3d(out3)
            f = f.transpose(1,2)
            g = self.norm1(f)
            # g = g.transpose(1,2)
            # print(f"f:{f}")
            # print(f"f.shape:{f.shape}")
            # print(f"x.shape:{x.shape}")
            # g = g.transpose(1,2)
            # print(f"f:{f}")
            # print(f"f.shape:{f.shape}")
            x = x + g
            # x = x + self.drop_path(self.mamba1(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is trbaue
            x = x + self.drop_path(self.mamba1(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module):

    def __init__(self,configs):
        super(TSLANet, self).__init__()
        self.configs = configs
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)
        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)
        
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i],configs1=self.configs)
            for i in range(args.depth)]
        )
        print(f"self.tsla_blocks:{self.tsla_blocks}")
        # Parameters/Embeddings
        # self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.pred_len)

    def pretrain(self, x_in):
        x = rearrange(x_in, 'b l m -> b m l')
        x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

        xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch]
        xb_mask = self.input_layer(xb_mask)

        for tsla_blk in self.tsla_blocks:
            xb_mask = tsla_blk(xb_mask)

        return xb_mask, self.input_layer(x_patched)


    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet(configs = args)
        print(f"self.model:{self.model}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        _, _, C = batch_x.shape
        batch_x = batch_x.float().to(device)

        preds, target = self.model.pretrain(batch_x)

        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet(configs=args)
        print(f"self.modle:{self.model}")
        self.criterion = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.preds = []
        self.trues = []
       
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                              verbose=True),
            'monitor': 'val_mse',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _calculate_loss(self, batch, mode="train"):
        batch_x, batch_y, _, _ = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -args.pred_len:, :]
        batch_y = batch_y[:, -args.pred_len:, :].to(device)
        loss = self.criterion(outputs, batch_y)

        pred = outputs.detach().cpu()
        true = batch_y.detach().cpu()

        mse = self.mse(pred.contiguous(), true.contiguous())
        mae = self.mae(pred, true)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, preds, trues = self._calculate_loss(batch, mode="test")
        self.preds.append(preds)
        self.trues.append(trues)
        return {'test_loss': loss, 'pred': preds, 'true': trues}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self):
        preds = torch.cat(self.preds)
        trues = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mse = self.mse(preds.contiguous(), trues.contiguous())
        mae = self.mae(preds, trues)
        print(f"{mae, mse}")


def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return model, pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        num_sanity_val_steps=0,
        devices=1,
        max_epochs=args.train_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(args.seed)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    mse_result = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
    mae_result = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

    return model, mse_result, mae_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Data args...
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/huhuajin/TSLANet/all_datasets/ETT-small/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # forecasting lengths
    parser.add_argument('--seq_len', type=int, default=256, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--n1',type=int,default=512,help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=128,help='Second Embedded representation')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--seed', type=int, default=51)

    # model
    parser.add_argument('--emb_dim', type=int, default=128, help='dimension of model')
    parser.add_argument('--depth', type=int, default=2, help='num of layers')
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout value')
    parser.add_argument('--patch_size', type=int, default=32, help='size of patches')
    parser.add_argument('--mask_ratio', type=float, default=0.4)

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=False)
    parser.add_argument('--Mamba', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)
    
    # METHOD
    parser.add_argument('--ch_ind', type=int, default=0, help='Channel Independence; True 1 False 0')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')
    parser.add_argument('--dconv1', type=int, default=1, help='d_conv parameter of Mamba')
    parser.add_argument('--dconv2', type=int, default=3, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
    parser.add_argument('--enc_in', type=int, default=321, help='encoder input size') #Use this hyperparameter as the number of channels
    # parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    # parser.add_argument('--embed', type=str, default='timeF',
                        # help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true',default=False,help='whether to predict unseen future data')

    args = parser.parse_args()

    device = torch.device('cuda:{}'.format(0))

    # load from checkpoint
    run_description = f"{args.data_path.split('.')[0]}_emb{args.emb_dim}_d{args.depth}_ps{args.patch_size}"
    run_description += f"_pl{args.pred_len}_bs{args.batch_size}_mr{args.mask_ratio}"
    run_description += f"_ASB_{args.ASB}_AF_{args.adaptive_filter}_ICB_{args.ICB}_preTr_{args.load_from_pretrained}"
    run_description += f"_{datetime.datetime.now().strftime('%H_%M')}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_mse',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')
    print("Dataset loaded ...")

    if args.load_from_pretrained:
        pretrained_model, best_model_path = pretrain_model()
    else:
        best_model_path = ''


    model, mse_result, mae_result = train_model(best_model_path)
    print("MSE results", mse_result)
    print("MAE  results", mae_result)

    # Save results into an Excel sheet ...
    df = pd.DataFrame({
        'MSE': mse_result,
        'MAE': mae_result
    })
    df.to_excel(os.path.join(CHECKPOINT_PATH, f"results_{datetime.datetime.now().strftime('%H_%M')}.xlsx"))

    # Append results into a text file ...
    os.makedirs("textOutput", exist_ok=True)
    f = open(f"textOutput/TSLANet_{os.path.basename(args.data_path)}.txt", 'a')
    f.write(run_description + "  \n")
    f.write('MSE:{}, MAE:{}'.format(mse_result, mae_result))
    f.write('\n')
    f.write('\n')
    f.close()
