import torch
import torch.nn as nn
import math

# position Embedding값
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 일반 self-attention은 토큰 길이가 제곱이라 비쌈 (L*L)
# Key, Value 길이 축을 토큰 길이를 Key의 길이로 줄여서 L*K만 계산
class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        # attention score(query|key)를 softmax전에 스케일링
        self.scale = (dim // n_heads) ** -0.5

        self.qw = nn.Linear(dim, dim, bias = bias)
        self.kw = nn.Linear(dim, dim, bias = bias)
        self.vw = nn.Linear(dim, dim, bias = bias)

        # 길이 축 축소용 파라미터
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))

        self.ow = nn.Linear(dim, dim, bias = bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        B, L, D = q.shape
        # MultiHead형태로 변경
        q = torch.reshape(q, [B, L, self.n_heads, -1])
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.reshape(k, [B, L, self.n_heads, -1])
        k = torch.permute(k, [0, 2, 3, 1])
        v = torch.reshape(v, [B, L, self.n_heads, -1])
        v = torch.permute(v, [0, 2, 3, 1])
        
        # 토큰축 길이를 L에서 k로 변경
        k = torch.matmul(k, self.E[:L, :])
        v = torch.matmul(v, self.F[:L, :])
        v = torch.permute(v, [0, 1, 3, 2])

        # q, k -> attention계산
        # softmax가 뾰족해지거나 불안정해지는 걸 줄이는 표준 attention 스케일링
        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1) # attention score

        v_attn = torch.matmul(attn, v)
        v_attn = torch.permute(v_attn, [0, 2, 1, 3])
        v_attn = torch.reshape(v_attn, [B, L, D])

        x = self.ow(v_attn)
        return x

# adaLN-zero (conditioning)
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# 
class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k, rate=0.0):
        super().__init__()
        # (B, L, D)일때 D를 정규화함
        self.ln_1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # self-attention
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln_2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(rate),
        )
        self.gamma_1 = nn.Linear(dim, dim) # scale_msa
        self.beta_1 = nn.Linear(dim, dim) # shift_msa
        self.gamma_2 = nn.Linear(dim, dim) # # scale_mlp
        self.beta_2 = nn.Linear(dim, dim) # shift_mlp
        self.scale_1 = nn.Linear(dim, dim) # gate_msa
        self.scale_2 = nn.Linear(dim, dim) # gate_mlp

        # Initialize weights to zero for modulation and gating layers
        self._init_weights([self.gamma_1, self.beta_1, self.gamma_2, 
                            self.beta_2, self.scale_1, self.scale_2])

    # adaLN-zero에서는 transformer진행할때 residual전에 
    # scale을 한번 곱하는데 처음 셋팅을 0으로 맞추고 함
    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    # x: 이미지, c: time embedding
    def forward(self, x, c):
        #c = self.ln_act(c)
        scale_msa = self.gamma_1(c)
        shift_msa = self.beta_1(c)
        scale_mlp = self.gamma_2(c)
        shift_mlp = self.beta_2(c)
        # residual하기전에 한번더 곱하는 변수
        gate_msa = self.scale_1(c).unsqueeze(1)
        gate_mlp = self.scale_2(c).unsqueeze(1)
        x = self.attn(modulate(self.ln_1(x), shift_msa, scale_msa)) * gate_msa + x
        return self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)) * gate_mlp + x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.ln_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

        self._init_weights([self.linear, self.gamma, self.beta])

    def _init_weights(self, layers):
        for layer in layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, c):
        scale = self.gamma(c)
        shift = self.beta(c)

        # adaLN-zero
        x = modulate(self.ln_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__( 
        self, img_size, 
        dim=64, patch_size=4, 
        depth=3, heads=4, 
        mlp_dim=512, k=64, 
        in_channels=3, 
    ):
        super(DiT, self).__init__()
        self.dim = dim
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, dim))
        self.patches = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, 
                      stride=patch_size, padding=0, bias=False),
        )
        
        self.transformer = nn.ModuleList()
        for i in range(self.depth):
            self.transformer.append(
                TransformerBlock(
                    self.n_patches, dim, heads, mlp_dim, k)
            )

        self.emb = nn.Sequential(
            PositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.final = FinalLayer(dim, patch_size, in_channels)
        # 채널을 공간 해상도로 재배치해서 업샘플링
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x, t):
        t = self.emb(t)
        x = self.patches(x)
        B, C, H, W = x.shape
        x = x.permute([0, 2, 3, 1]).reshape([B, H * W, C])
        x += self.pos_embedding

        for layer in self.transformer:
            x = layer(x, t)

        x = self.final(x, t).permute([0, 2, 1])
        x = x.reshape([B, -1, H, W])
        x = self.ps(x)
        return x