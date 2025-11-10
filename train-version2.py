import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

# ============================================================================
# DÁN CÁC CLASS `DecoderBlock` và `MiniChatGPT` CỦA BẠN VÀO ĐÂY
# Giả sử file `decoderblock_GPT.py` tồn tại và chứa class DecoderBlock
# Hoặc bạn có thể dán trực tiếp class đó vào đây.
# Ở đây tôi sẽ tạo một class giả để code có thể chạy được.
# ============================================================================

class DecoderBlock(nn.Module):
    """
    Đây là một class DecoderBlock giả. 
    Hãy thay thế nó bằng class thật của bạn từ file `decoderblock_GPT.py`.
    """
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, activation="swish"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU() if activation != "swish" else nn.SiLU(), # Swish is SiLU
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Tạo causal mask
        seq_len = x.size(1)
        device = x.device
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Self-attention
        attn_output, _ = self.self_attn(
            x, x, x, 
            attn_mask=causal_mask, 
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class MiniChatGPT(nn.Module):
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
        self.decoder_component = nn.ModuleList([
            DecoderBlock(embed_dim=self.embed_dim, num_heads=num_heads, 
                         ffn_hidden_dim=ffn_hidden_dim, dropout=dropout, activation="swish")
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(embed_dim)
        # Output
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False) # GPT-2 không dùng bias ở đây
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Proper weight initialization as per GPT-2 paper"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 1. Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)
        
        x = token_emb + pos_emb
        x = self.dropout_embed(x)
        
        # 2. Transformer Decoder Blocks
        # attention_mask dùng để che đi các padding token
        padding_mask = (input_ids == self.pad_token_id) if attention_mask is None else (attention_mask == 0)

        for decoder_layer in self.decoder_component:
            x = decoder_layer(x, key_padding_mask=padding_mask)
            
        # 3. Final normalization and language model head
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
    
    def count_parameters(self):
        """Đếm tổng số parameters có thể huấn luyện"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================================
# 1. CẤU HÌNH (CONFIGURATION)
# ============================================================================
class Config:
    # Model params
    vocab_size = 10000     # Kích thước bộ từ vựng (giả định)
    max_seq_len = 256      # Chiều dài chuỗi tối đa
    embed_dim = 256        # Kích thước embedding
    num_heads = 4          # Số lượng attention heads
    num_layers = 4         # Số lượng decoder blocks
    ffn_hidden_dim = 1024  # 4 * embed_dim
    dropout = 0.1
    pad_token_id = 1       # ID của padding token
    
    # Training params
    batch_size = 32
    num_epochs = 5
    learning_rate = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "best_model.pth"
    
# ============================================================================
# 2. DỮ LIỆU (DATA)
# ============================================================================
class RandomTextDataset(Dataset):
    """
    Một Dataset giả để tạo ra các câu ngẫu nhiên.
    Trong thực tế, bạn sẽ dùng tokenizer để xử lý văn bản thật.
    """
    def __init__(self, num_samples, max_len, vocab_size):
        self.num_samples = num_samples
        self.max_len = max_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Tạo một chuỗi token ngẫu nhiên
        random_sequence = torch.randint(2, self.vocab_size, (self.max_len,)) # Bắt đầu từ 2 để tránh 0 (unk) và 1 (pad)
        
        # Mục tiêu của mô hình là dự đoán token tiếp theo
        # input_ids:  [t_1, t_2, ..., t_n-1]
        # target_ids: [t_2, t_3, ..., t_n]
        input_ids = random_sequence[:-1]
        target_ids = random_sequence[1:]
        
        return input_ids, target_ids

# ============================================================================
# 3. HÀM HUẤN LUYỆN (TRAINING FUNCTION)
# ============================================================================
def train(config):
    print(f"🚀 Starting training on {config.device}...")
    
    # a. Khởi tạo Dataset và DataLoader
    train_dataset = RandomTextDataset(num_samples=1000, max_len=config.max_seq_len + 1, vocab_size=config.vocab_size)
    val_dataset = RandomTextDataset(num_samples=200, max_len=config.max_seq_len + 1, vocab_size=config.vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # b. Khởi tạo Model, Optimizer, Loss Function
    model = MiniChatGPT(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ffn_hidden_dim=config.ffn_hidden_dim,
        dropout=config.dropout,
        pad_token_id=config.pad_token_id
    ).to(config.device)
    
    print(f"Model created with {model.count_parameters():,} trainable parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # ignore_index=pad_token_id để loss function bỏ qua các vị trí padding
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # c. Vòng lặp huấn luyện
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        # -- Training Phase --
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]")
        for input_ids, target_ids in progress_bar:
            input_ids, target_ids = input_ids.to(config.device), target_ids.to(config.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids) # Shape: [batch_size, seq_len, vocab_size]

            # Calculate loss
            # CrossEntropyLoss yêu cầu logits có shape [N, C] và target có shape [N]
            # Ta cần reshape logits và targets
            loss = criterion(logits.view(-1, config.vocab_size), target_ids.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            # Gradient clipping để tránh bùng nổ gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # -- Validation Phase --
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]")
            for input_ids, target_ids in progress_bar_val:
                input_ids, target_ids = input_ids.to(config.device), target_ids.to(config.device)
                
                logits = model(input_ids)
                loss = criterion(logits.view(-1, config.vocab_size), target_ids.view(-1))
                total_val_loss += loss.item()
                progress_bar_val.set_postfix(loss=loss.item())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # d. Lưu checkpoint nếu có kết quả tốt hơn
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🎉 New best model found! Saving checkpoint to {config.checkpoint_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config
            }, config.checkpoint_path)
            
    print("✅ Training finished!")

# ============================================================================
# 4. CHẠY CHƯƠNG TRÌNH
# ============================================================================
if __name__ == "__main__":
    config = Config()
    train(config)