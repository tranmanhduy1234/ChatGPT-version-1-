"""
DIAGNOSE & RETRAIN MINI CHAT GPT
Kiểm tra vấn đề và retrain với config tối ưu
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import wandb
from model import MiniChatGPT
from dataset import ViLQADataset

class ImprovedTrainer:
    """Improved Trainer with W&B Logging"""
    
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device,
                 output_dir="./outputs", num_epochs=10, use_wandb=True, project_name="mini-chatGPT"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.best_val_loss = float('inf')
        self.patience = 3
        self.patience_counter = 0
        self.global_step = 0
    
    def train(self):
        """Main training loop"""
        if self.use_wandb:
            wandb.init(
                project=wandb.config.get("project") if hasattr(wandb, 'config') else "mini-chatgpt",
                reinit=True
            )
        
        try:
            for epoch in range(self.num_epochs):
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
                print(f"{'='*60}")
                
                # quá trình train
                train_loss = self._train_epoch(epoch)
                val_loss = self._validate(epoch)
                
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "learning_rate": current_lr,
                    })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, is_best=True)
                    print("✅ Best model saved")
                    
                    if self.use_wandb:
                        wandb.log({"best_val_loss": val_loss})
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\n⚠️  Early stopping after {self.patience} epochs without improvement")
                        break
        
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device) # [1, 1, 1, 1, ..., 0, 0, 0] => 0 là padding
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log batch loss to W&B
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/global_step": self.global_step,
                })
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def _validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        path = self.output_dir / (f"best_model.pt" if is_best else f"checkpoint-{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }, path)
        print(f"   Saved: {path}")
        
        # Log to W&B
        if self.use_wandb and is_best:
            wandb.save(str(path))

# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🚀 RETRAIN MINI CHAT GPT - With WANDB Logging")
    print("="*60 + "\n")
    
    # CONFIG - OPTIMIZED
    CONFIG = {
        "max_length": 512,
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 0.05,
        "embed_dim": 768,
        "num_heads": 8,
        "num_layers": 12,
        "ffn_hidden_dim": 3072,
        "dropout": 0.3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": "./outputs/vilqa_chatgpt_v2",
        "num_train_samples": None,
        "num_val_samples": None,
        "use_wandb": True,
        "wandb_project": "mini-chatgpt-vilqa"
    }
    
    print(f"📊 Config:\n{json.dumps(CONFIG, indent=2)}\n")
    
    # ===== Initialize W&B =====
    if CONFIG["use_wandb"]:
        print("📊 Initializing Weights & Biases...\n")
        wandb.init(
            project=CONFIG["wandb_project"],
            config=CONFIG,
            reinit=True
        )
    
    # ===== Load Dataset =====
    print("1️⃣  Loading Dataset...")
    ds = load_dataset("huyhuy123/ViLQA")
    train_data = ds["train"]
    train_val = train_data.train_test_split(test_size=0.1, seed=42)
    
    train_data = train_val["train"]
    if CONFIG["num_train_samples"]:
        train_data = train_data.select(range(min(CONFIG["num_train_samples"], len(train_data))))
    
    val_data = train_val["test"]
    if CONFIG["num_val_samples"]:
        val_data = val_data.select(range(min(CONFIG["num_val_samples"], len(val_data))))
    
    print(f"   ✓ Train: {len(train_data)}, Val: {len(val_data)}\n")
    
    # ===== Load Tokenizer =====
    print("2️⃣  Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    except:
        print('##### Lỗi load tokenizer')
        exit(0)
    
    vocab_size = len(tokenizer)
    print(f"   ✓ Vocab size: {vocab_size}\n")
    
    # ===== Create Datasets =====
    print("3️⃣  Creating Datasets...")
    train_dataset = ViLQADataset(train_data, tokenizer, CONFIG["max_length"], "train")
    val_dataset = ViLQADataset(val_data, tokenizer, CONFIG["max_length"], "validation")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    print(f"   ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # ===== Create Model =====
    print("4️⃣  Creating Model...")
    model = MiniChatGPT(
        vocab_size=vocab_size,
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        ffn_hidden_dim=CONFIG["ffn_hidden_dim"],
        dropout=CONFIG["dropout"],
        pad_token_id=tokenizer.pad_token_id
    )
    
    params = model.count_parameters()
    print(f"   ✓ Parameters: {params/1e6:.2f}M\n")
    
    # ===== Setup Training =====
    print("5️⃣  Setup Training...")
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    trainer = ImprovedTrainer(
        model, train_loader, val_loader, optimizer, scheduler,
        CONFIG["device"], CONFIG["output_dir"], CONFIG["num_epochs"],
        use_wandb=CONFIG["use_wandb"], project_name=CONFIG["wandb_project"]
    )
    print("   ✓ Ready\n")
    
    # ===== Train =====
    print("6️⃣  Starting Training...\n")
    trainer.train()
    
    print(f"\n{'='*60}")
    print("✅ Training Complete!")
    print(f"📁 Saved to: {CONFIG['output_dir']}")
    if CONFIG["use_wandb"]:
        print(f"📊 W&B Project: {CONFIG['wandb_project']}")
    print("="*60)

if __name__ == "__main__":
    main()