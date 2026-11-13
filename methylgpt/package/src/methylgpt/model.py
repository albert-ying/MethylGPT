"""MethylGPT network definitions (inference copy of the research code).

PPCACore: closed-form probabilistic PCA of 49,156 CpGs; posterior factor scores use only the
observed CpGs of each profile. MethylGPTNet: dense-input transformer encoder over the profile and
its factor scores, with per-CpG readouts for reconstruction. Heads/BioHeads: age, sex, tissue
and 25 distilled full-array clock outputs from the 2,048-dimensional embedding.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPCACore(torch.nn.Module):
    """x = mu + V s + e, s_k ~ N(0, lam_k), e ~ N(0, sigma2); V orthonormal over all CpGs.

    The posterior mean of s uses only the observed CpGs of each profile. Because V^T V = I,
    V_O^T V_O = V^T diag(w) V for the 0/1 observation weights w.
    """

    def __init__(self, mu, V, lam, sigma2, rank=None):
        super().__init__()
        rank = rank or V.shape[1]
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("V", torch.as_tensor(V[:, :rank], dtype=torch.float32).contiguous())
        self.register_buffer("lam", torch.as_tensor(lam[:rank], dtype=torch.float32))
        self.sigma2 = float(sigma2)
        self.rank = rank

    @classmethod
    def load(cls, path: Path, rank=None):
        z = np.load(path)
        rank = rank or int(z["V"].shape[1])
        trace, lam_all = float(z["trace"]), z["lam"].astype(np.float64)
        sigma2 = (trace - lam_all[:rank].sum()) / (z["V"].shape[0] - rank)
        return cls(z["mu"], z["V"], z["lam"], sigma2, rank)

    @torch.no_grad()
    def posterior(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x: (B, P) float, any value where w == 0; w: (B, P) {0,1}. Returns s (B, K)."""
        xc = torch.where(w > 0, x - self.mu, torch.zeros((), device=x.device)) * w
        b = (xc @ self.V) / self.sigma2
        prior = torch.diag(1.0 / self.lam)
        s = torch.empty(x.shape[0], self.rank, device=x.device)
        for i in range(x.shape[0]):
            G = (self.V * w[i, :, None]).T @ self.V
            A = G / self.sigma2 + prior
            L = torch.linalg.cholesky(A)
            s[i] = torch.cholesky_solve(b[i, :, None], L)[:, 0]
        return s

    @torch.no_grad()
    def predict(self, s: torch.Tensor, cols=None) -> torch.Tensor:
        if cols is None:
            return self.mu + s @ self.V.T
        return self.mu[cols] + s @ self.V[cols].T


class Attn(nn.Module):
    def __init__(self, d, heads, kv_dim=None):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(kv_dim or d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, ctx, key_valid=None):
        B, Lq, D = x.shape
        Lk = ctx.shape[1]
        q = self.q(x).view(B, Lq, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, Lk, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if key_valid is None:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Explicit attention: PyTorch's memory-efficient kernel is very slow in the backward
            # pass for few queries over ~49k masked keys (profiled at ~0.5 s per step).
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(D // self.h)
            scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
            y = torch.matmul(torch.softmax(scores, -1).to(v.dtype), v)
        return self.o(y.transpose(1, 2).reshape(B, Lq, D))


class Block(nn.Module):
    def __init__(self, d, heads, kv_dim=None, cross=False, mlp=4):
        super().__init__()
        self.cross = cross
        self.n1 = nn.LayerNorm(d)
        self.nc = nn.LayerNorm(kv_dim or d) if cross else None
        self.attn = Attn(d, heads, kv_dim if cross else None)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, ctx=None, key_valid=None):
        h = self.n1(x)
        x = x + (self.attn(h, self.nc(ctx), key_valid) if self.cross else self.attn(h, h))
        return x + self.ff(self.n2(x))


class MethylGPTNet(nn.Module):
    def __init__(self, core: PPCACore, d_tok=256, d=512, heads=8, n_latent=128, n_score_tokens=16,
                 n_self=8, n_fourier=8, c_scale=0.2, d_read=64, encoder="tokens", d_dense=2048,
                 n_dense_tokens=32):
        super().__init__()
        self.core = core
        self.encoder_type = encoder
        P, K = core.V.shape
        self.P, self.K = P, K
        self.c_scale = c_scale
        self.e_scale = math.sqrt(core.sigma2)
        self.id_emb = nn.Parameter(torch.randn(P, d_tok) * 0.02)
        self.load_c = nn.Parameter(torch.randn(P, d_tok) / math.sqrt(d_tok))
        self.load_e = nn.Parameter(torch.randn(P, d_tok) / math.sqrt(d_tok))
        self.register_buffer("freq", torch.arange(1, n_fourier + 1, dtype=torch.float32) * math.pi)
        self.val = nn.Sequential(nn.Linear(3 + 2 * n_fourier, d_tok), nn.GELU(), nn.Linear(d_tok, d_tok))
        self.tok_norm = nn.LayerNorm(d_tok)
        self.tok_mlp = nn.Sequential(nn.Linear(d_tok, 2 * d_tok), nn.GELU(), nn.Linear(2 * d_tok, d_tok))
        self.latents = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.n_score_tokens = n_score_tokens
        self.score_proj = nn.Linear(K, n_score_tokens * d)
        self.enc_cross = Block(d, heads, kv_dim=d_tok, cross=True)
        if encoder == "dense":
            # Dense input layer over the whole profile (denoising-autoencoder style): every weight
            # receives gradient from every profile, unlike one attention read over ~49k tokens.
            self.n_dense_tokens = n_dense_tokens
            self.dense_c = nn.Linear(P, d_dense, bias=False)
            self.dense_e = nn.Linear(P, d_dense, bias=False)
            self.dense_w = nn.Linear(P, d_dense)
            self.dense_mlp = nn.Sequential(nn.GELU(), nn.LayerNorm(d_dense), nn.Linear(d_dense, d_dense), nn.GELU(),
                                           nn.Linear(d_dense, n_dense_tokens * d))
        self.enc_self = nn.ModuleList([Block(d, heads) for _ in range(n_self)])
        self.q_emb = nn.Parameter(torch.randn(P, d) * 0.02)
        self.q_load = nn.Linear(K, d, bias=False)
        self.dec1 = Block(d, heads, cross=True)
        self.dec2 = Block(d, heads, cross=True)
        self.sum_norm = nn.LayerNorm(d)
        self.h_norm = nn.LayerNorm(d)
        self.sum_proj = nn.Linear(d, d_read)
        self.h_proj = nn.Linear(d, d_read)
        # Low-dimensional per-CpG readout vectors, zero-initialised: training starts at the core.
        self.u1 = nn.Parameter(torch.zeros(P, d_read))
        self.u2 = nn.Parameter(torch.zeros(P, d_read))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # Score correction: the transformer adjusts the core's factor scores (in prior-SD units).
        # It has no per-CpG parameters, so its gradient pools over every target of a profile.
        self.ds_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, K))
        nn.init.zeros_(self.ds_head[-1].weight)
        nn.init.zeros_(self.ds_head[-1].bias)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @torch.no_grad()
    def inputs(self, x, ctx, s, fit=None):
        """fp32 token features computed outside autocast. x: (B, P) betas with NaN where
        unobserved; ctx: (B, P) bool context positions; s: (B, K) core scores in the full
        core's basis; fit: optional (B, P) reconstruction from an out-of-fold core
        (cross-fitted training). Without it the full core's reconstruction is used."""
        mu = self.core.mu
        xs = torch.where(ctx, x, mu)  # placeholder at excluded positions (masked from attention)
        if fit is None:
            fit = self.core.predict(s)
        c = (xs - mu) / self.c_scale
        e = torch.where(ctx, xs - fit, torch.zeros((), device=x.device)) / self.e_scale
        return xs, c, e

    def encode(self, xs, c, e, ctx, s):
        if self.encoder_type == "dense":
            return self.encode_dense(c, e, ctx, s)
        f = xs[..., None] * self.freq
        feats = torch.cat([xs[..., None], c[..., None], e[..., None], torch.sin(f), torch.cos(f)], -1)
        tok = self.id_emb + self.val(feats) + c[..., None] * self.load_c + e[..., None] * self.load_e
        tok = tok + self.tok_mlp(self.tok_norm(tok))
        sz = s / torch.sqrt(self.core.lam)
        score_tokens = self.score_proj(sz).view(xs.shape[0], self.n_score_tokens, -1)
        z = torch.cat([self.latents.expand(xs.shape[0], -1, -1).to(score_tokens.dtype), score_tokens], 1)
        z = self.enc_cross(z, tok, ctx)
        for blk in self.enc_self:
            z = blk(z)
        return z

    def encode_dense(self, c, e, ctx, s):
        w = ctx.to(c.dtype)
        scale = (self.P / w.sum(1, keepdim=True).clamp_min(1.0)).to(c.dtype)  # inverted-dropout scaling
        h = self.dense_c(c * w * scale) + self.dense_e(e * scale) + self.dense_w(w)
        dense_tokens = self.dense_mlp(h).view(c.shape[0], self.n_dense_tokens, -1)
        sz = s / torch.sqrt(self.core.lam)
        score_tokens = self.score_proj(sz).view(c.shape[0], self.n_score_tokens, -1)
        z = torch.cat([self.latents.expand(c.shape[0], -1, -1).to(score_tokens.dtype), score_tokens,
                       dense_tokens.to(score_tokens.dtype)], 1)
        for blk in self.enc_self:
            z = blk(z)
        return z

    def embedding(self, z, s):
        with torch.autocast(z.device.type, enabled=False):
            return torch.cat([s.float() / torch.sqrt(self.core.lam), self.sum_norm(z.float().mean(1))], -1)

    def residual(self, z, tgt_idx):
        """tgt_idx: (B, T) CpG indices. Returns (B, T) residual predictions."""
        lam_sqrt = torch.sqrt(self.core.lam)
        q = self.q_emb[tgt_idx] + self.q_load((self.core.V[tgt_idx] * lam_sqrt).to(z.dtype))
        h = self.dec1(q, z)
        h = self.dec2(h, z)
        summ = self.sum_proj(self.sum_norm(z.mean(1)))
        lin = (self.u1[tgt_idx] * summ[:, None, :]).sum(-1)
        ctxr = (self.u2[tgt_idx] * self.h_proj(self.h_norm(h))).sum(-1)
        return lin + ctxr + self.out(h).squeeze(-1)

    def base(self, s, tgt_idx):
        """fp32 core prediction at target CpGs (call outside autocast)."""
        return self.core.mu[tgt_idx] + torch.einsum("btk,bk->bt", self.core.V[tgt_idx], s)

    def score_correction(self, z):
        return self.ds_head(z.mean(1)).float() * torch.sqrt(self.core.lam)

    def predict_targets(self, z, s, tgt_idx, amp_dtype=torch.bfloat16, fit=None):
        """Full prediction at target CpGs given encoder output z and core scores s."""
        with torch.autocast(z.device.type, dtype=amp_dtype, enabled=z.is_cuda):
            r = self.residual(z, tgt_idx)
            ds = self.score_correction(z)
        with torch.autocast(z.device.type, enabled=False):
            base = self.base(s, tgt_idx) if fit is None else fit.gather(1, tgt_idx)
            corr = torch.einsum("btk,bk->bt", self.core.V[tgt_idx], ds.float())
        return base + corr + r.float(), base

    def forward(self, x, ctx, s, tgt_idx, amp_dtype=torch.bfloat16, fit=None):
        with torch.autocast(x.device.type, enabled=False):
            xs, c, e = self.inputs(x, ctx, s, fit)
        with torch.autocast(x.device.type, dtype=amp_dtype, enabled=x.is_cuda):
            z = self.encode(xs, c, e, ctx, s)
        pred, base = self.predict_targets(z, s, tgt_idx, amp_dtype, fit)
        return pred, base, z


class Heads(nn.Module):
    def __init__(self, d_in, n_tissue):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 1024), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(1024, 512), nn.GELU())
        self.age = nn.Linear(512, 1)
        self.sex = nn.Linear(512, 1)
        self.tissue = nn.Linear(512, n_tissue)

    def forward(self, e):
        h = self.trunk(e)
        return self.age(h).squeeze(-1), self.sex(h).squeeze(-1), self.tissue(h)


class BioHeads(Heads):
    """Aging heads plus a regression head for distilled full-array biomarkers."""

    def __init__(self, d_in, n_tissue, n_bio):
        super().__init__(d_in, n_tissue)
        self.bio = nn.Linear(512, n_bio)

    def forward(self, e):
        h = self.trunk(e)
        return self.age(h).squeeze(-1), self.sex(h).squeeze(-1), self.tissue(h), self.bio(h)
