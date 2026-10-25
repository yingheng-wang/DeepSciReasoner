import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Sci_Reasoner.transformer import Decoder, MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer


# ==========================================
# Velocity Field Decoder: f_θ(x, t; C)
# x = continuous energy coordinates sampled from [a,b]
# t = time
# C = h_glob (global crystal embedding)
# ==========================================
class VelocityNet(nn.Module):
    def __init__(self, args):
        super(VelocityNet, self).__init__()
        
        self.label_dim = args.label_dim
        self.dec_layers = args.dec_layers
        self.dec_dropout = args.dec_dropout
        self.d_model = args.d_model
        self.h = args.h
        self.d_ff = args.d_ff
        self.t_dim = 1
        # crystal_ctx dimension = 2 * d_model (from GAT: pooled_nodes + embedded_composition)
        self.ctx_dim = args.d_model * 2

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.h, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dec_dropout)

        # Input: x (1) + t (1) + crystal_ctx (ctx_dim) → d_model
        self.fc = nn.Linear(1 + self.t_dim + self.ctx_dim, self.d_model)
        self.decoder1 = Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout), self.dec_layers)
        self.decoder2 = Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dec_dropout), self.dec_layers)

        self.generator = nn.Linear(self.d_model, 1)


    def forward(self, x_coords, t, crystal_ctx):
        """
        f_θ(x, t; C) — Algorithm 1 Line 17/26
        
        Args:
            x_coords: [bs, N] — spatial coordinates (sampled during training, grid during inference)
            t: [bs] or scalar — current time
            crystal_ctx: [bs, ctx_dim] — global crystal embedding C = h_glob
        Returns:
            velocity field: [bs, N]
        """
        if isinstance(t, float) or (isinstance(t, torch.Tensor) and t.ndim == 0):
            t_vec = torch.ones(crystal_ctx.shape[0], 1).to(crystal_ctx.device) * t
        else:
            t_vec = t.view(-1, 1)  # [bs, 1]

        bs = crystal_ctx.shape[0]
        seq_len = x_coords.shape[1]

        # Directly concatenate raw x, t, and crystal context C
        x_raw = x_coords.unsqueeze(-1)                                 # [bs, N, 1]
        ctx_repeated = crystal_ctx.unsqueeze(1).repeat(1, seq_len, 1)  # [bs, N, ctx_dim]
        t_repeated = t_vec.unsqueeze(1).repeat(1, seq_len, 1)          # [bs, N, 1]
        
        dec_input = torch.cat((x_raw, t_repeated, ctx_repeated), dim=-1)  # [bs, N, 1+1+ctx_dim]
        dec_input = F.leaky_relu(self.fc(dec_input))  # [bs, N, d_model]

        # Self-attention (flag=True skips cross-attention)
        out = self.decoder1(dec_input, dec_input, None, None, flag=True) 
        out = self.decoder2(out, out, None, None, flag=True)

        out = self.generator(out).squeeze(-1)  # [bs, N]

        return out


# ==========================================
# MIR Model — Mass-Conserving Iterative Refinement
# Matches Algorithm 1 (Stage II + Inference)
# ==========================================
class MIR(nn.Module):
    def __init__(self, args):
        super(MIR, self).__init__()
        self.args = args
        
        if self.args.data_src == 'binned_dos_256':
            self.x_min = -5.0
            self.x_max = 5.0
        elif self.args.data_src in ['binned_dos_128', 'ph_dos_51', 'raman_128_sigma10']:
            self.x_min = -4.0
            self.x_max = 4.0
        elif self.args.data_src in ['xanes_128_sigma0.5', 'xanes_O_128_sigma0.5']:
            self.x_min = -5.0
            self.x_max = 40.0
        else:
            raise ValueError(f'No MIR spectral grid defined for data_src={self.args.data_src!r}')

        self.num_points = args.label_dim
        grid = torch.linspace(self.x_min, self.x_max, self.num_points)
        self.register_buffer('grid', grid)
        self.dx = (self.x_max - self.x_min) / (self.num_points - 1)

        self.net = VelocityNet(args)
        self.criterion = torch.nn.MSELoss()

    def pred_v(self, x_coords, t, crystal_ctx):
        """
        Predict velocity field: f_θ(x, t; C)
        Matches Algorithm 1 Line 17/26
        """
        return self.net(x_coords, t, crystal_ctx)

    def sample_timestep(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def compute_reference_velocity(self, p0, pT, t, x_coords):
        """
        Build OT displacement-interpolation reference velocity field f*(x,t)
        and evaluate at sampled x coordinates. (Algorithm 1, Lines 13-15)
        
        Args:
            p0, pT: [bs, N_grid] — normalized distributions (sum=1, mass space)
            t: [bs, 1] — sampled time
            x_coords: [bs, N_sample] — sampled spatial coordinates
        Returns:
            vt_target: [bs, N_sample] — reference velocity at sampled points
        """
        batch_size = p0.shape[0]
        device = p0.device
        
        # Compute CDFs (cumsum of mass gives valid CDF, no dx needed)
        cdf0 = torch.cumsum(p0, dim=1)
        cdfT = torch.cumsum(pT, dim=1)

        # Safety normalization
        cdf0 = cdf0 / cdf0[:, -1:].clamp(min=1e-6)
        cdfT = cdfT / cdfT[:, -1:].clamp(min=1e-6)

        vt_list = []
        
        cdf0_np = cdf0.detach().cpu().numpy()
        cdfT_np = cdfT.detach().cpu().numpy()
        grid_np = self.grid.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy().flatten()
        x_np = x_coords.detach().cpu().numpy()
        
        s_grid = np.linspace(0, 1, self.num_points)
        
        for i in range(batch_size):
            ti = t_np[i]
            
            # Compute quantile functions Q_0(s), Q_T(s) via inverse CDF
            x0_s = np.interp(s_grid, cdf0_np[i], grid_np)
            xT_s = np.interp(s_grid, cdfT_np[i], grid_np)
            
            # Displacement interpolation: x(t,s) = (1-t)Q_0(s) + tQ_T(s)
            xt_s = (1 - ti) * x0_s + ti * xT_s
            
            # Reference velocity: f*(x(t,s),t) = Q_T(s) - Q_0(s)
            v_particle = xT_s - x0_s
            
            if xt_s[-1] <= xt_s[0]:
                xt_s[-1] = xt_s[0] + 1e-6
                
            # Convert to Eulerian form at SAMPLED x coordinates
            vt_at_x = np.interp(x_np[i], xt_s, v_particle, left=0, right=0)
            
            vt_list.append(vt_at_x)
            
        vt_tensor = torch.tensor(np.array(vt_list), dtype=torch.float32).to(device)
        
        return vt_tensor

    def forward(self, x_mass, y_mass, crystal_ctx):
        """
        Training forward pass (Algorithm 1, Lines 10-18)
        
        Args:
            x_mass: [bs, N] — predicted p0 from Stage I (sum=1)
            y_mass: [bs, N] — ground truth pT (sum=1)
            crystal_ctx: [bs, ctx_dim] — C = h_glob from frozen encoder
        """
        bs = x_mass.shape[0]
        device = x_mass.device
        t = self.sample_timestep(bs, device)
        t_expanded = t.view(-1, 1)

        # Sample x from Unif([a,b]) (Algorithm 1 Line 16)
        x_sampled = torch.rand(bs, self.num_points, device=device) * (self.x_max - self.x_min) + self.x_min
        x_sampled, _ = torch.sort(x_sampled, dim=1)  # sort for stable processing

        # Build reference velocity at sampled points (Algorithm 1 Lines 13-15)
        with torch.no_grad():
            vt_target = self.compute_reference_velocity(x_mass, y_mass, t_expanded, x_sampled)

        # Predict velocity: f_θ(x, t; C) at sampled points (Algorithm 1 Line 17)
        v_pred = self.pred_v(x_sampled, t, crystal_ctx)

        loss = self.criterion(v_pred, vt_target)
        return loss

    @torch.no_grad()
    def inference(self, p0_mass, crystal_ctx, dt=None, num_steps=100):
        """
        Inference (Algorithm 1, Lines 20-32)
        
        Args:
            p0_mass: [bs, N] — initial prediction from Stage I (sum=1)
            crystal_ctx: [bs, ctx_dim] — C = h_glob
        """
        batch_size = p0_mass.shape[0]
        device = p0_mass.device
        
        if dt is None:
            dt = 1.0 / num_steps
            
        # Grid coordinates for inference (Algorithm 1 Line 26: evaluate on grid)
        x_grid = self.grid.unsqueeze(0).repeat(batch_size, 1)  # [bs, N]
            
        # Initialize CDF from mass
        cdf = torch.cumsum(p0_mass, dim=1)
        cdf = cdf / cdf[:, -1:].clamp(min=1e-6)
        
        for k in range(num_steps):
            t_curr = k * dt
            
            # Evaluate velocity on grid: v_i ← f_θ(x_i, t_k; C) (Line 26)
            t_tensor = torch.tensor([t_curr], device=device).repeat(batch_size)
            velocity = self.pred_v(x_grid, t_tensor, crystal_ctx)
            
            velocity[:, 0] = 0.0
            velocity[:, -1] = 0.0
            
            # Upwind CDF update (Eq. 22): F^{k+1}_i = F^k_i - Δt * v_i * D_x^up F^k_i
            diff_left = torch.zeros_like(cdf)
            diff_left[:, 0] = 0
            diff_left[:, 1:] = (cdf[:, 1:] - cdf[:, :-1]) / self.dx
            
            diff_right = torch.zeros_like(cdf)
            diff_right[:, :-1] = (cdf[:, 1:] - cdf[:, :-1]) / self.dx
            diff_right[:, -1] = 0
            
            mask_pos = (velocity > 0).float()
            dF_dx_up = mask_pos * diff_left + (1 - mask_pos) * diff_right
            
            cdf = cdf - dt * velocity * dF_dx_up
            cdf = torch.clamp(cdf, 0.0, 1.0)
            cdf = torch.cummax(cdf, dim=1)[0]

        
        # Recover
        cdf_padded = F.pad(cdf, (1, 0), mode='constant', value=0)
        p_final = cdf_padded[:, 1:] - cdf_padded[:, :-1]


        return p_final