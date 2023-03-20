import torch
from torch import nn
from torch.nn import functional as F


# 残差块
class EncoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_out // 4
        self.id_path = nn.Conv2d(self.dim_in, self.dim_out, 1) if self.dim_in != self.dim_out else nn.Identity()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim_in, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_out, 1)
        )

    def forward(self, x):
        return self.id_path(x) + self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_out // 4
        self.id_path = nn.Conv2d(self.dim_in, self.dim_out, 1) if self.dim_in != self.dim_out else nn.Identity()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim_in, self.dim_hid, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_hid, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim_hid, self.dim_out, 3, 1, 1)
        )

    def forward(self, x):
        return self.id_path(x) + self.block(x)


##############################  VAE  #################################
class Encoder(nn.Module):
    def __init__(self, in_dim=3, dim=64):
        super().__init__()
        self.in_dim = in_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, dim, 7, padding=3),
            EncoderBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2),
            EncoderBlock(dim, dim),
            nn.MaxPool2d(kernel_size=2),
            EncoderBlock(dim, 2 * dim),
            nn.MaxPool2d(kernel_size=2),
            EncoderBlock(2 * dim, 4 * dim),
            nn.ReLU(),
        )
        # self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3,stride=1, padding=1)
        # self.conv2 = EncoderBlock(dim, dim)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = EncoderBlock(dim, dim)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv4 = EncoderBlock(dim, 2 * dim)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # self.conv5 = EncoderBlock(2 * dim, 4 * dim)
        # self.relu = nn.ReLU()


    def forward(self, x):
        return self.encoder(x)
        # print("forward")
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.maxpool1(x)
        # print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        # x = self.maxpool2(x)
        # print(x.shape)
        # x = self.conv4(x)
        # print(x.shape)
        # x = self.maxpool3(x)
        # print(x.shape)
        # x = self.conv5(x)
        # print(x.shape)
        # x = self.relu(x)
        # return x


class Decoder(nn.Module):
    """Decoder of VQ-VAE"""

    def __init__(self, out_dim=3, dim=64):
        super().__init__()
        self.out_dim = out_dim

        self.decoder = nn.Sequential(
            DecoderBlock(4 * dim, 2 * dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            DecoderBlock(2 * dim, dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            DecoderBlock(dim, dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            DecoderBlock(dim, dim),
            nn.ReLU(),
            nn.Conv2d(dim, out_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x, get_idx=False):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)
        # (BHW, )
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, H, W, C]

        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            if get_idx:
                return quantized, encoding_indices
            return quantized

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


class VQVAE(nn.Module):
    """VQ-VAE"""
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance=1, commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance

        self.encoder = Encoder(in_dim, embedding_dim//4)
        self.vq_layer = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = Decoder(in_dim, embedding_dim//4)

    def forward(self, x):
        z = self.encoder(x)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon

        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)

        recon_loss = F.mse_loss(x_recon, x) / self.data_variance

        return e_q_loss + recon_loss