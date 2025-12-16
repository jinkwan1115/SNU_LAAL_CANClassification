import torch
import torch.nn as nn
import numpy as np

class PatchTimeSeriesEmbedding(nn.Module):
    """
    시계열 데이터를 패치(Patch) 단위로 자르고 임베딩하는 클래스
    (최신 TSFM인 TimesFM 등의 입력 처리 방식)
    """
    def __init__(self, patch_size, in_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        # 패치 하나를 벡터로 투영 (Linear Projection)
        self.projection = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x):
        # x shape: [Batch, In_Channels, Time_Length]
        B, C, L = x.shape
        
        # 1. Padding to fit patch size
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len))
            L = x.shape[-1]

        # 2. Patching: [B, C, L] -> [B, Num_Patches, Patch_Size * C]
        # (구현 편의를 위해 채널을 flatten하여 패치에 포함시킵니다)
        num_patches = L // self.patch_size
        x = x.view(B, C, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, num_patches, -1)
        
        # 3. Projection: [B, Num_Patches, d_model]
        x = self.projection(x)
        return x

class PretrainedFoundationModel(nn.Module):
    """
    사전 학습된 시계열 파운데이션 모델 (예: Chronos, Moirai 등)을 모사한 클래스.
    """
    def __init__(self, d_model=128, n_head=4, n_layers=4, patch_size=16, in_channels=1):
        super().__init__()
        self.patch_embed = PatchTimeSeriesEmbedding(patch_size, in_channels, d_model)
        
        # Transformer Encoder (Backbone)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.d_model = d_model

    def forward(self, x):
        """
        Input: [Batch, Channels, Time]
        Output: [Batch, Num_Patches, d_model] (Contextual Representations)
        """
        # 1. Patch Embedding
        embeddings = self.patch_embed(x)
        
        # 2. Pass through Frozen Transformer
        # (이 부분의 가중치는 학습되지 않음)
        features = self.encoder(embeddings)
        
        return features

def load_foundation_model(device='cpu'):
    """
    모델을 로드하고 파라미터를 Freeze(동결)하여 반환하는 함수
    """
    model = PretrainedFoundationModel()
    model.to(device)
    model.eval() # 학습 모드 해제
    
    # 3차년도 핵심: Foundation Model 자체는 학습하지 않음 (Freeze)
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"[Info] Pre-trained Foundation Model loaded on {device}. Parameters are frozen.")
    return model