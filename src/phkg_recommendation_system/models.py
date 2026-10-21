import torch
import torch.nn as nn

class TimeEncoder(nn.Module):
    """16-dimensional learned cosine encoding of elapsed time."""
    def __init__(self, time_dim=16):
        super().__init__()
        self.linear = nn.Linear(1, time_dim)

    def forward(self, delta_t):
        delta_t = delta_t.float().view(-1, 1)
        return torch.cos(self.linear(delta_t))

class TGN3BiMem(nn.Module):
    """
    Architecture used for both TGN-3 and TGN-All.
    The difference is the amount of pre-target history supplied.
    Memory is re-initialised for each prediction window.
    """
    def __init__(self, num_graph_protocols, num_output_protocols,
                 memory_dim=32, time_dim=16, dropout=0.2):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_graph_protocols = num_graph_protocols
        self.num_output_protocols = num_output_protocols
        self.patient_initial = nn.Parameter(torch.zeros(memory_dim))
        self.protocol_base = nn.Embedding(num_graph_protocols, memory_dim)
        self.time_encoder = TimeEncoder(time_dim)

        self.patient_message = nn.Sequential(
            nn.Linear(memory_dim * 3 + time_dim, memory_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        self.protocol_message = nn.Sequential(
            nn.Linear(memory_dim * 3 + time_dim, memory_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(memory_dim, memory_dim)
        )
        self.patient_updater = nn.GRUCell(memory_dim, memory_dim)
        self.protocol_updater = nn.GRUCell(memory_dim, memory_dim)

        self.scorer = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(memory_dim, 1)
        )

    def initialise_memories(self, device):
        patient_memory = self.patient_initial.clone()
        protocol_memory = torch.zeros(
            self.num_graph_protocols, self.memory_dim,
            dtype=torch.float32, device=device
        )
        return patient_memory, protocol_memory

    def process_episode(self, patient_memory, protocol_memory,
                        protocol_ids, delta_t, protocol_to_graph_idx):
        device = patient_memory.device
        graph_idx = torch.tensor(
            [protocol_to_graph_idx[p] for p in protocol_ids],
            dtype=torch.long, device=device
        )
        old_patient = patient_memory
        old_protocol_memory = protocol_memory[graph_idx]
        protocol_base = self.protocol_base(graph_idx)

        n = len(graph_idx)
        patient_expanded = old_patient.unsqueeze(0).expand(n, -1)
        delta = torch.full((n,), float(delta_t),
                           dtype=torch.float32, device=device)
        time_emb = self.time_encoder(delta)

        message_input = torch.cat(
            [patient_expanded, protocol_base,
             old_protocol_memory, time_emb], dim=1
        )

        patient_messages = self.patient_message(message_input)
        aggregated_patient_message = patient_messages.mean(dim=0)
        protocol_messages = self.protocol_message(message_input)

        new_patient = self.patient_updater(
            aggregated_patient_message.unsqueeze(0),
            old_patient.unsqueeze(0)
        ).squeeze(0)

        new_protocol_states = self.protocol_updater(
            protocol_messages, old_protocol_memory
        )

        updated_protocol_memory = protocol_memory.clone()
        updated_protocol_memory[graph_idx] = new_protocol_states

        return new_patient, updated_protocol_memory

    def score_protocols(self, patient_memory, protocol_memory,
                        output_graph_indices):
        base = self.protocol_base(output_graph_indices)
        temporal = protocol_memory[output_graph_indices]
        candidate_repr = base + temporal

        patient = patient_memory.unsqueeze(0).expand(
            len(output_graph_indices), -1
        )
        interaction = patient * candidate_repr
        difference = torch.abs(patient - candidate_repr)

        score_input = torch.cat(
            [patient, candidate_repr, interaction, difference], dim=1
        )
        return self.scorer(score_input).squeeze(-1)

class GRURecommender(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=32,
                 output_dim=51, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.output(self.dropout(hidden[-1]))

class LSTMRecommender(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=32,
                 output_dim=51, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.output(self.dropout(hidden[-1]))

class TransformerRecommender(nn.Module):
    def __init__(self, input_dim=51, d_model=32, nhead=2,
                 num_layers=1, dim_feedforward=64, output_dim=51,
                 dropout=0.2, sequence_length=3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, activation="relu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x) + self.position_embedding
        x = self.encoder(x)
        return self.output(self.dropout(x[:, -1, :]))
