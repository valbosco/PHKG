from dataclasses import dataclass
import os
import torch

@dataclass(frozen=True)
class Config:
    seed:int=42
    episode_gap_seconds:int=60
    min_episodes:int=4
    n_output_protocols:int=51
    memory_dim:int=32
    time_dim:int=16
    dropout:float=0.2
    learning_rate:float=1e-3
    weight_decay:float=1e-4
    max_epochs:int=100
    patience:int=10
    grad_clip:float=1.0
    device:str="cuda" if torch.cuda.is_available() else "cpu"
    @property
    def neo4j_uri(self): return os.getenv("NEO4J_URI","bolt://localhost:7687")
    @property
    def neo4j_user(self): return os.getenv("NEO4J_USER","")
    @property
    def neo4j_password(self): return os.getenv("NEO4J_PASSWORD","")
