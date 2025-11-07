import torch
import torch.nn as nn

def diagnose_checkpoint(checkpoint_path):
    """Kiểm tra checkpoint có vấn đề gì"""
    print("\n" + "="*60)
    print("🔍 DIAGNOSING CHECKPOINT")
    print("="*60 + "\n")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print("📋 Checkpoint Contents:")
    for key in checkpoint.keys():
        if key == "model_state_dict":
            state = checkpoint[key]
            print(f"\n   {key}:")
            for name, param in list(state.items())[:3]:
                print(f"      - {name}: shape={param.shape}, dtype={param.dtype}")
                print(f"        mean={param.mean():.4f}, std={param.std():.4f}")
        elif key == "history":
            hist = checkpoint[key]
            print(f"\n   {key}:")
            if hist.get("train_loss"):
                print(f"      - Train loss: {hist['train_loss'][-5:]}")
            if hist.get("val_loss"):
                print(f"      - Val loss: {hist['val_loss']}")
        else:
            print(f"\n   {key}: {checkpoint[key]}")
    
    # Check if weights are extreme
    state = checkpoint["model_state_dict"]
    for name, param in state.items():
        if torch.isnan(param).any():
            print(f"\n⚠️  NaN detected in {name}")
        if torch.isinf(param).any():
            print(f"\n⚠️  Inf detected in {name}")
        if param.std() < 1e-5:
            print(f"\n⚠️  Near-zero std in {name}: {param.std()}")
# ============================================================================
# 1. MINI CHAT GPT MODEL (từ training script)
# ============================================================================
'''
Sửa lại khối decoder cho phù hợp với mô hình chatGPT -không có lớp cross-attention
'''
class MiniChatGPT(nn.Module):
    """Improved Model"""
    
    def __init__(self, vocab_size, max_seq_len=512, embed_dim=512, num_heads=8, 
                 num_layers=6, ffn_hidden_dim=2048, dropout=0.2, pad_token_id=1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout_embed = nn.Dropout(dropout)
        
        # Decoder with better initialization
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # ← Better stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm_final = nn.LayerNorm(embed_dim)
        
        # Output
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=True)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        _, seq_len = input_ids.shape
        
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)
        
        x = token_emb + pos_emb
        x = self.dropout_embed(x)
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1
        ) # dạng
        # torch.triu(..., diagonal=1) → 
        # [[0, 1, 1, 1],
        # [0, 0, 1, 1],
        # [0, 0, 0, 1],
        # [0, 0, 0, 0]]  # True là 1 (masked), False là 0
        
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        x = self.decoder(x, x, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        x = self.norm_final(x)
        
        logits = self.lm_head(x)
        return logits
    
    def count_parameters(self):
        """Đếm tổng số parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)