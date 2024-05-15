import diffusers
import torch
import torch.nn as nn



class VariationalAutoencoder(nn.Module):
    def __init__(self, pretrained_path):
        super().__init__()
        # 로컬 경로에서 모델 로드
        self.model = diffusers.AutoencoderKL.from_pretrained(pretrained_path, use_safetensors=True)
        self.model.requires_grad_(False)  # 모델의 파라미터를 고정
        self.model.enable_slicing()  # 모델의 일부만 사용 가능하도록 설정 (메모리 절약)

    @torch.no_grad()  # 그라디언트 계산 비활성화
    def encode(self, x):
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        return z

    @torch.no_grad()  # 그라디언트 계산 비활성화
    def decode(self, z):
        z = 1. / self.model.scaling_factor * z
        x = self.model.decode(z).sample
        return x
