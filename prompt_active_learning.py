import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from foundation_encoder_loader import load_foundation_model

class PromptTuningWrapper(nn.Module):
    """
    [핵심 구조]
    Frozen Foundation Model + Learnable Prompt Vectors + Classifier Head
    """
    def __init__(self, foundation_model, num_classes, num_prompts=10):
        super().__init__()
        self.foundation_model = foundation_model
        self.d_model = foundation_model.d_model
        
        # 1. Learnable Soft Prompts (학습 가능한 프롬프트 벡터)
        # [1, Num_Prompts, d_model]
        self.soft_prompts = nn.Parameter(torch.randn(1, num_prompts, self.d_model))
        
        # 2. Classification Head (Lightweight)
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, num_classes)
        )

    def forward(self, x):
        # 1. 입력 시계열을 패치 임베딩으로 변환 (From Frozen Model)
        # x_embed: [Batch, Num_Patches, d_model]
        x_embed = self.foundation_model.patch_embed(x)
        
        # 2. 프롬프트 붙이기 (Prefix Tuning / Prompt Tuning)
        batch_size = x.shape[0]
        # prompts: [Batch, Num_Prompts, d_model]
        prompts = self.soft_prompts.expand(batch_size, -1, -1)
        
        # 입력 앞에 프롬프트 결합: [Batch, Num_Prompts + Num_Patches, d_model]
        combined_input = torch.cat((prompts, x_embed), dim=1)
        
        # 3. Foundation Model 통과 (Backbone)
        # features: [Batch, Seq_Len, d_model]
        features = self.foundation_model.encoder(combined_input)
        
        # 4. Pooling (여기서는 Global Average Pooling 사용)
        # 시계열 전체 문맥을 요약
        pooled_features = torch.mean(features, dim=1)
        
        # 5. Classification
        logits = self.classifier(pooled_features)
        return logits

class ActiveLearningAgent:
    """
    능동 학습(Active Learning)을 수행하는 에이전트
    - 데이터 풀 관리 (Labeled vs Unlabeled)
    - 쿼리 전략 (Uncertainty Sampling)
    - 모델 업데이트 (Prompt Learning)
    """
    def __init__(self, model, unlabeled_data, device='cpu'):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # 데이터 관리 (Tensor 형태로 가정)
        self.unlabeled_pool = unlabeled_data # Tensor [N, C, L]
        self.labeled_data = None             # Tensor
        self.labeled_targets = None          # Tensor
        
        # 마스킹 배열 (True면 Labeled, False면 Unlabeled)
        self.is_labeled = np.zeros(len(unlabeled_data), dtype=bool)

    def query_samples(self, n_query=5):
        """
        [Query Strategy]
        가장 불확실한(Entropy가 높은) 샘플의 인덱스를 반환
        """
        self.model.eval()
        unlabeled_indices = np.where(~self.is_labeled)[0]
        
        if len(unlabeled_indices) == 0:
            return []

        # Unlabeled 데이터에 대한 예측 수행
        dataset = TensorDataset(self.unlabeled_pool[unlabeled_indices])
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        
        all_probs = torch.cat(all_probs)
        
        # Entropy 계산: -sum(p * log(p))
        entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-6), dim=1)
        
        # Entropy가 가장 높은 상위 n_query개 선택
        top_k_indices = torch.topk(entropy, k=min(n_query, len(entropy))).indices
        query_indices = unlabeled_indices[top_k_indices.numpy()]
        
        print(f"[Active Learning] Selected {len(query_indices)} samples with highest uncertainty.")
        return query_indices

    def update_labels(self, indices, new_labels):
        """
        사용자(Oracle)로부터 받은 레이블을 데이터셋에 추가
        """
        # 마킹 업데이트
        self.is_labeled[indices] = True
        
        new_data = self.unlabeled_pool[indices]
        new_targets = torch.tensor(new_labels, dtype=torch.long)
        
        if self.labeled_data is None:
            self.labeled_data = new_data
            self.labeled_targets = new_targets
        else:
            self.labeled_data = torch.cat((self.labeled_data, new_data))
            self.labeled_targets = torch.cat((self.labeled_targets, new_targets))

    def train_step(self, epochs=5):
        """
        확보된 Labeled 데이터로 프롬프트와 헤드만 학습
        """
        if self.labeled_data is None:
            print("[Warning] No labeled data to train.")
            return

        self.model.train()
        dataset = TensorDataset(self.labeled_data, self.labeled_targets)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 간단한 로그 출력
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# --- 실행 예시 (Main Logic) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 가상 데이터 생성 (Batch: 100, Channel: 1, Time: 128)
    # 실제 연구에서는 dataloader_preprocessing.py를 통해 로드
    dummy_data = torch.randn(100, 1, 128).to(device)
    dummy_labels = torch.randint(0, 3, (100,)).numpy() # 3개 클래스 (정답지 - 실제론 숨겨짐)
    
    # 2. 파운데이션 모델 로드 (Frozen)
    foundation_model = load_foundation_model(device)
    
    # 3. 프롬프트 튜닝 모델 초기화
    # 3개의 클래스로 분류, 10개의 프롬프트 토큰 사용
    prompt_model = PromptTuningWrapper(foundation_model, num_classes=3, num_prompts=10).to(device)
    
    # 파라미터 체크: 프롬프트와 Classifier만 학습 가능한지 확인
    trainable_params = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in prompt_model.parameters())
    print(f"[Info] Trainable Params: {trainable_params} / Total Params: {total_params}")
    
    # 4. 능동 학습 에이전트 시작
    agent = ActiveLearningAgent(prompt_model, dummy_data, device)
    
    # [Cycle 1] 초기 쿼리 (랜덤 혹은 불확실성 기반)
    query_idx = agent.query_samples(n_query=10)
    
    # 사용자 레이블링 시뮬레이션 (실제로는 GUI나 입력을 받음)
    user_labels = dummy_labels[query_idx] 
    agent.update_labels(query_idx, user_labels)
    
    # 학습 수행
    print("\n--- Start Training Cycle 1 ---")
    agent.train_step(epochs=10)
    
    # [Cycle 2] 모델이 헷갈리는 데이터 추가 쿼리
    query_idx_2 = agent.query_samples(n_query=5)
    user_labels_2 = dummy_labels[query_idx_2]
    agent.update_labels(query_idx_2, user_labels_2)
    
    # 재학습 (점진적 향상)
    print("\n--- Start Training Cycle 2 ---")
    agent.train_step(epochs=10)
    
    print("\n[Done] Active Prompt Learning Completed.")