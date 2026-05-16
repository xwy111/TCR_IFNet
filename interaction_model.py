import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.weight_norm import weight_norm
from transformers import T5EncoderModel


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.num_grids = num_grids
        if denominator is None:
            self.denominator = (grid_max - grid_min) / (num_grids - 1)
        else:
            self.denominator = denominator

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int, grid_min: float = -2.,
                 grid_max: float = 2., num_grids: int = 8,
                 use_base_update: bool = True,
                 use_layernorm: bool = True,
                 base_activation=F.silu,
                 spline_weight_init_scale: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.layernorm is not None:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class AttentionWithFastKANTransform(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, head_dim: int, num_heads: int, gating: bool = True):
        super(AttentionWithFastKANTransform, self).__init__()
        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating

        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        self.norm = head_dim ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # q: [Batch, Q_len, Dim]
        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)

        att = (wq * wk).sum(-1)
        if bias is not None:
            att = att + bias

        att = att.softmax(-2)
        del wq, wk

        wv = self.linear_v(v).view(*v.shape[:-2], 1, v.shape[-2], self.num_heads, -1)
        # 加权求和
        o = (att[..., None] * wv).sum(-3)
        del att, wv

        o = o.view(*o.shape[:-2], -1)

        if self.linear_g is not None:
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        o = self.linear_o(o)
        return o

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0.2):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=4):
        super(BANLayer, self).__init__()
        self.c = 32
        self.k = k
        self.h_out = h_out
        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)
        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        v_ = self.v_net(v)
        q_ = self.q_net(q)

        if self.h_out <= self.c:
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            # 备用分支，一般不会走到这
            v_t = v_.transpose(1, 2).unsqueeze(3)
            q_t = q_.transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_t, q_t)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i

        logits = self.bn(logits)
        return logits, att_maps

class GatedCNN(nn.Module):
    def __init__(self, dim_model, kernel_size=5, n_layers=3, dropout=0.2):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.dim_model = dim_model
        self.scale = math.sqrt(0.5)
        self.convs = nn.ModuleList([
            nn.Conv1d(dim_model, 2 * dim_model, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_model)
        self.proj = nn.Linear(dim_model, dim_model)

    def forward(self, x):

        conv_input = self.proj(x).permute(0, 2, 1)
        for conv in self.convs:
            residual = conv_input
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + residual) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class BCFM_KAN(nn.Module):
    def __init__(self, dim_model=256, dropout=0.2):
        super().__init__()

        self.local_encoder_p = GatedCNN(dim_model, kernel_size=3, n_layers=2, dropout=dropout)
        self.local_encoder_t = GatedCNN(dim_model, kernel_size=3, n_layers=2, dropout=dropout)

        self.num_heads = 4
        assert dim_model % self.num_heads == 0, "dim_model must be divisible by num_heads"
        head_dim = dim_model // self.num_heads

        self.cross_attn = AttentionWithFastKANTransform(
            q_dim=dim_model, k_dim=dim_model, v_dim=dim_model,
            head_dim=head_dim, num_heads=self.num_heads, gating=True
        )

        self.norm_p = nn.LayerNorm(dim_model)
        self.norm_t = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def _make_additive_bias(self, mask, dtype):

        bias = torch.zeros_like(mask, dtype=dtype)
        bias = bias.masked_fill(~mask.bool(), -1e9)

        return bias.unsqueeze(1).unsqueeze(-1)

    def forward(self, x_p, x_t, mask_p, mask_t):

        p_loc = self.local_encoder_p(x_p)
        t_loc = self.local_encoder_t(x_t)
        x_p = x_p + p_loc
        x_t = x_t + t_loc

        bias_p = self._make_additive_bias(mask_p, x_p.dtype)  # 用于 mask Peptide
        bias_t = self._make_additive_bias(mask_t, x_t.dtype)  # 用于 mask TCR

        t_induced = self.cross_attn(q=x_t, k=x_p, v=x_p, bias=bias_p)
        p_induced = self.cross_attn(q=x_p, k=x_t, v=x_t, bias=bias_t)

        x_t = self.norm_t(x_t + self.dropout(t_induced))
        x_p = self.norm_p(x_p + self.dropout(p_induced))

        return x_p, x_t

class ProTCR_InducedFit_Final(nn.Module):
    def __init__(self, model_path="Rostlab/prot_t5_xl_uniref50", project_dim=256, unfreeze_last_n_layers=12):
        super().__init__()
        print(f">> Initializing ProTCR with Linear+PReLU Decouple, KAN-Attention & BAN ...")

        # 1. LLM Backbone (ProtT5)
        self.encoder = T5EncoderModel.from_pretrained(model_path)
        self.encoder.gradient_checkpointing_enable()
        self.encoder.enable_input_require_grads()

        # 冻结与解冻策略
        for param in self.encoder.parameters(): param.requires_grad = False
        total_layers = len(self.encoder.encoder.block)
        for i in range(total_layers - unfreeze_last_n_layers, total_layers):
            for param in self.encoder.encoder.block[i].parameters(): param.requires_grad = True
        for param in self.encoder.encoder.final_layer_norm.parameters(): param.requires_grad = True

        # 2. Decouple Layer (Linear + PReLU)
        # 稳健且省显存
        self.decouple_p = nn.Sequential(
            nn.Linear(1024, 1024), nn.PReLU(),
            nn.Linear(1024, project_dim), nn.PReLU()
        )
        self.decouple_t = nn.Sequential(
            nn.Linear(1024, 1024), nn.PReLU(),
            nn.Linear(1024, project_dim), nn.PReLU()
        )

        # 3. KAN 驱动的序列交互模块
        self.bcfm = BCFM_KAN(dim_model=project_dim)

        # 4. 双线性注意力网络 (BAN)
        self.ban_layer = BANLayer(v_dim=project_dim, q_dim=project_dim, h_dim=project_dim, h_out=2, k=3)

        # 5. Classifier (Linear + PReLU)
        self.classifier = nn.Sequential(
            nn.Linear(project_dim, 128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

        self._init_submodules()

    def _init_submodules(self):
        # 为 Linear 层进行初始化, PReLU 自适应
        for m in [self.decouple_p, self.decouple_t, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None: nn.init.constant_(layer.bias, 0)
        # KAN 和 BAN 内部已有初始化逻辑，无需在此重复

    def compute_robust_focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (alpha * (1 - pt) ** gamma * ce_loss).mean()

    def forward(self, pep_ids, pep_mask, tcr_ids, tcr_mask, labels=None):
        # 1. 编码 (Encoder)
        p_out = self.encoder(input_ids=pep_ids, attention_mask=pep_mask).last_hidden_state
        t_out = self.encoder(input_ids=tcr_ids, attention_mask=tcr_mask).last_hidden_state

        # 2. 降维 (Decouple)
        p_proj = self.decouple_p(p_out)
        t_proj = self.decouple_t(t_out)

        p_mask_expanded = pep_mask.unsqueeze(-1).float()
        t_mask_expanded = tcr_mask.unsqueeze(-1).float()

        p_input_feat = (p_proj * p_mask_expanded).sum(dim=1) / torch.clamp(p_mask_expanded.sum(dim=1), min=1e-9)
        t_input_feat = (t_proj * t_mask_expanded).sum(dim=1) / torch.clamp(t_mask_expanded.sum(dim=1), min=1e-9)

        input_feat = torch.cat([p_input_feat, t_input_feat], dim=-1)

        y_p, y_t = self.bcfm(p_proj, t_proj, pep_mask, tcr_mask)

        ban_feat, att_maps = self.ban_layer(y_p, y_t)

        logits = self.classifier(ban_feat)

        loss = None
        if labels is not None:
            loss = self.compute_robust_focal_loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "att_maps": att_maps,
            "input_feat": input_feat,
            "latent_feat": ban_feat
        }