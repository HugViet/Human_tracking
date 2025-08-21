import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2

class ResNet18Embedder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(base.children())[:-1]  # remove fc
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(512, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        feats = self.backbone(x)  # Bx512x1x1
        feats = torch.flatten(feats, 1)
        feats = self.fc(feats)
        feats = F.normalize(feats, p=2, dim=1)
        return feats

class FeatureExtractor:
    def __init__(self, device='cpu', out_dim=128, input_size=(256, 128)):
        self.device = torch.device(device)
        self.input_size = input_size  # (H, W) for person crops
        self.model = ResNet18Embedder(out_dim=out_dim).to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.input_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def __call__(self, frame, detections_xyxy):
        if len(detections_xyxy) == 0:
            return np.empty((0, self.model.out_dim), dtype=np.float32)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        crops = []
        h, w = frame.shape[:2]
        for box in detections_xyxy:
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            x1, y1 = min(x1, w - 1), min(y1, h - 1)
            x2, y2 = min(x2, w - 1), min(y2, h - 1)
            if x2 <= x1 or y2 <= y1:
                # invalid box → dummy crop
                crops.append(torch.zeros(3, *self.input_size))
                continue

            crop = rgb[y1:y2, x1:x2].copy() 
            tensor = self.tf(crop)
            crops.append(tensor)

        batch = torch.stack(crops).to(self.device)
        feats = self.model(batch).cpu().numpy().astype(np.float32)
        return feats

