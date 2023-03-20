import torch
from torch import nn
import math


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None, need_weights=False):
        attn_mask = attn_mask.to(dtype=q.dtype, device=q.device) if attn_mask is not None else None
        x, att_weights = self.attn(self.ln_q(q), self.ln_kv(k), self.ln_kv(v), need_weights=need_weights, attn_mask=attn_mask)
        x = q + self.dropout(x)
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x, att_weights


class PositionEncode(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=600):
        super(PositionEncode, self).__init__()
        self.dp = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dp(x)


class TransformerAudioHead(nn.Module):
    def __init__(self, transformer_width: int, transformer_layers: int, transformer_heads: int,
                 output_dim: int, dropout: float = 0.1):
        super(TransformerAudioHead, self).__init__()

        LayerClass = nn.TransformerEncoderLayer
        _layer = LayerClass(
            transformer_width,
            transformer_heads,
            dim_feedforward=transformer_width * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=20, stride=10),
            nn.InstanceNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=14, stride=7, padding=3),
            nn.InstanceNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=14, stride=7, padding=3),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=6, stride=3, padding=2),
            nn.InstanceNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, transformer_width, kernel_size=1, stride=1)
        )
        self.pos_emd = PositionEncode(transformer_width)
        self.transformer = nn.TransformerEncoder(_layer, transformer_layers)

        self.layer_norm = nn.LayerNorm(transformer_width, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ln_audio_final = nn.LayerNorm(transformer_width)
        self.autio_projection = nn.Linear(transformer_width, output_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x):
        # x: batch x L x D
        x = self.encoder(x.permute([0, 2, 1])).permute([0, 2, 1])
        x = self.pos_emd(x.transpose(0, 1)).transpose(0, 1)
        x = self.layer_norm(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(
            x,
            mask=None,
            src_key_padding_mask=None,
        )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_audio_final(x)

        x = self.autio_projection(x)

        return x


class TransformerFirstImageVQHead(nn.Module):
    def __init__(self, transformer_width: int, transformer_layers: int, transformer_heads: int,
                 output_dim: int, dropout: float = 0.1):
        super(TransformerFirstImageVQHead, self).__init__()
        LayerClass = nn.TransformerEncoderLayer
        _layer = LayerClass(
            transformer_width,
            transformer_heads,
            dim_feedforward=transformer_width * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.pos_emd = PositionEncode(transformer_width)
        self.transformer = nn.TransformerEncoder(_layer, transformer_layers)

        self.layer_norm = nn.LayerNorm(transformer_width, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ln_vq_final = nn.LayerNorm(transformer_width)
        self.vq_projection = nn.Linear(transformer_width, output_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, x):
        # x: batch x C x H x W
        batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]
        x = x.permute([0, 2, 3, 1]).view(batch_size, h*w, -1)
        x = self.pos_emd(x.transpose(0, 1)).transpose(0, 1)
        x = self.layer_norm(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(
            x,
            mask=None,
            src_key_padding_mask=None,
        )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_vq_final(x)

        x = self.vq_projection(x)

        return x


class AutoVoDecoder(nn.Module):
    def __init__(self):
        super(AutoVoDecoder, self).__init__()
        self.self_attn_1 = TransformerBlock(256, 4, dropout=0.0)
        self.cross_attn_1 = TransformerBlock(256, 4, dropout=0.0)
        self.self_attn_2 = TransformerBlock(256, 4, dropout=0.0)
        self.cross_attn_2 = TransformerBlock(256, 4, dropout=0.0)
        self.self_attn_3 = TransformerBlock(256, 4, dropout=0.0)
        self.cross_attn_3 = TransformerBlock(256, 4, dropout=0.0)

    def forward(self, audio_latent, img_latent):
        audio_latent, _ = self.self_attn_1(audio_latent, audio_latent, audio_latent, need_weights=True)
        audio_latent, _ = self.cross_attn_1(audio_latent, img_latent, img_latent, need_weights=True)
        audio_latent, _ = self.self_attn_2(audio_latent, audio_latent, audio_latent, need_weights=True)
        audio_latent, _ = self.cross_attn_2(audio_latent, img_latent, img_latent, need_weights=True)
        audio_latent, _ = self.self_attn_3(audio_latent, audio_latent, audio_latent, need_weights=True)
        audio_latent, _ = self.cross_attn_3(audio_latent, img_latent, img_latent, need_weights=True)
        return audio_latent


class AutoVo(nn.Module):
    def __init__(self):
        super(AutoVo, self).__init__()
        self.audio_encoder = TransformerAudioHead(transformer_width=256, transformer_layers=2, transformer_heads=4, output_dim=256, dropout=0.0)
        # 特征图的h和w是15和27
        # self.H_positional_embedding = nn.Parameter(scale * torch.randn(1, 15, 1, 256))
        # self.W_positional_embedding = nn.Parameter(scale * torch.randn(1, 1, 27, 256))
        self.vq_encoder = TransformerFirstImageVQHead(transformer_width=256, transformer_layers=2, transformer_heads=4, output_dim=256, dropout=0.0)
        self.decoder = AutoVoDecoder()
        self.layer_norm = nn.LayerNorm(256, eps=1e-8, elementwise_affine=True)
        self.fc = nn.Linear(256, 15*27 * 512)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, audio, imgs, vae):
        # batch x L x 2, batch x L x 3 x h x w
        batch_size, L, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        with torch.no_grad():
            # (batch_sizexL) x c x h/8 x w/8
            # (batch_sizexLxh/8xw/8, )
            x, codebook_idx = vae.vq_layer(vae.encoder(imgs.view(batch_size*L, 3, h, w)), True)
            # batch_size x L x c x h/8 x w/8
            x = x.view(batch_size, L, -1, h//8, w//8)
        # batch_size x L x h/8 x w/8 x c
        x = x.permute([0, 1, 3, 4, 2]).contiguous()
        # x = x + self.H_positional_embedding + self.W_positional_embedding
        # batch_size x h/8 x w/8 x c
        first_imgs = x[:, 0, :, :, :]
        C = first_imgs.shape[3]
        first_imgs = self.vq_encoder(first_imgs.permute([0, 3, 1, 2]).contiguous())
        first_imgs = first_imgs.view(batch_size, -1, C).permute(1, 0, 2).contiguous()  # (h/8 x w/8) x batch_size x C

        # batch_size x L x C
        audio = self.audio_encoder(audio)
        audio = audio.permute([1, 0, 2]).contiguous()  # L x batch_size x C

        out = self.decoder(audio, first_imgs)
        # batch_size x L（一秒30） x C, C=15*27 * 512
        out = out.permute([1, 0, 2]).contiguous()
        out = self.layer_norm(out)
        # 音频的秒数是视频的十倍
        L = out.shape[1] // 10
        out = out[:, :L, :]
        out = self.fc(out.view(-1, 256)).view(batch_size, -1, 512)

        out = out.view(batch_size, -1, 15*27, 512)
        out = out.view(-1, 512)
        if not self.training:
            codebook_idx = out.argmax(dim=1)
            vq = vae.vq_layer.quantize(codebook_idx)
            vq = vq.view(batch_size*L, 15, 27, 256).permute(0, 3, 1, 2).contiguous()
            # (batch_size x L) x 3 x h x w
            recon = vae.decoder(vq)
            return recon.view(batch_size, L, 3, h, w)
        loss = self.loss_fn(out, codebook_idx)
        return loss