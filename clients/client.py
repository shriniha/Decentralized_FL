import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import socket
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.cnn_model import CNNModel
from utils.data_loader import SkinCancerDataset

# Globals
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LR = 0.00001
PROXIMAL_MU = 0.1  # FedProx-like term for heterogeneity

# LocalClient class (P2P version—no gRPC/server)
class LocalClient:
    def __init__(self, cid):
        self.cid = cid
        self.model = CNNModel().to(DEVICE)
        self.train_ds = SkinCancerDataset(f"clients_data/{cid}", "train")
        self.test_ds = SkinCancerDataset(f"clients_data/{cid}", "test")
        self.train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size=BATCH_SIZE, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-4)

    def get_params(self):
        return [val.cpu().numpy().astype(np.float32) for val in self.model.state_dict().values()]

    def set_params(self, params):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), params):
            state_dict[key] = torch.tensor(val).to(DEVICE)
        self.model.load_state_dict(state_dict)

    def train_epoch(self, global_params=None):
        if global_params:
            self.set_params(global_params)
        self.model.train()
        for images, labels in self.train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            if global_params is not None:
                # FedProx proximal term
                proximal_loss = 0
                for p_local, p_global in zip(self.model.parameters(), global_params):
                    proximal_loss += (p_local - torch.tensor(p_global).to(DEVICE)) ** 2
                loss += (PROXIMAL_MU / 2) * proximal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.optimizer.step()
            print(f"[Client {self.cid}] Batch loss: {loss.item():.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.get_params()

    def evaluate(self):
        self.model.eval()
        correct, total, loss_total = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        avg_loss = loss_total / total if total > 0 else 0
        accuracy = correct / total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return avg_loss, accuracy