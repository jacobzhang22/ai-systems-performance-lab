import torch
import torch.nn as nn


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.lm_head(h)