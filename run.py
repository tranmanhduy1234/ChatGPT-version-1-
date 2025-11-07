"""
LOAD VÀ SỬ DỤNG TRAINED MINI CHAT GPT MODEL
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
from model import MiniChatGPT

class ChatGPTInference:
    """Class để load model và generate text"""
    def __init__(self, checkpoint_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"📱 Device: {device}\n")
        # ===== Load Tokenizer =====
        print("1️⃣  Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            print("   ✓ Loaded PhoBERT tokenizer\n")
        except:
            print("   ⚠️  Using fallback tokenizer...\n")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
        vocab_size = len(self.tokenizer)
        # ===== Load Checkpoint =====
        print("2️⃣  Loading checkpoint...")
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"   ✓ Loaded: {checkpoint_path}")
        
        # Extract model state dict
        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            print(f"   Keys in checkpoint: {checkpoint.keys()}")
        else:
            model_state = checkpoint
        # ===== Create Model =====
        print("\n3️⃣  Creating model...")
        self.model = MiniChatGPT(
            vocab_size=vocab_size,
            max_seq_len=256,
            embed_dim=512,
            num_heads=8,
            num_layers=6,
            ffn_hidden_dim=2048,
            dropout=0.1,
            pad_token_id=self.tokenizer.pad_token_id
        )
        print('Padding token id', self.tokenizer.pad_token_id)
        # Load state dict
        self.model.load_state_dict(model_state)
        self.model = self.model.to(device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✓ Model loaded: {total_params / 1e6:.2f}M parameters\n")
        
        # ===== Print Config =====
        if "epoch" in checkpoint:
            print(f"📊 Checkpoint Info:")
            print(f"   - Epoch: {checkpoint['epoch']}")
            print(f"   - Global step: {checkpoint.get('global_step', 'N/A')}")
            if "history" in checkpoint:
                hist = checkpoint["history"]
                if hist.get("val_loss"):
                    print(f"   - Final Val Loss: {hist['val_loss'][-1]:.4f}")
                if hist.get("train_loss"):
                    print(f"   - Final Train Loss: {hist['train_loss'][-1]:.4f}")
            print()
    
    def generate(
        self,
        prompt,
        max_length=150,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        use_argmax=False
    ):
        """
        Generate text từ prompt
        Args:
            prompt: str - prompt đầu vào
            max_length: int - độ dài max output
            temperature: float - điều chỉnh độ "sáng tạo" (0.1-2.0)
            top_k: int - chỉ xem xét top-k tokens
            top_p: float - nucleus sampling (0-1)
            use_argmax: bool - dùng argmax thay vì sampling (deterministic)
        
        Returns:
            str - generated text
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        original_length = input_ids.shape[1]
        
        print(f"\n🎯 Generating from prompt: \"{prompt}\"")
        print(f"   (Input length: {original_length} tokens)\n")
        
        generated_tokens = []
        
        with torch.no_grad():
            for i in range(max_length):
                # Forward pass
                outputs = self.model(input_ids)
                
                # Get logits của token cuối cùng
                next_token_logits = outputs[0, -1, :]
                
                # Debug: in ra top tokens
                if i < 3:  # Chỉ in 3 token đầu
                    top_probs, top_indices = torch.topk(torch.softmax(next_token_logits, dim=-1), k=5)
                    top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                    print(f"   Step {i+1} - Top 5 tokens: {list(zip(top_tokens, top_probs.tolist()))}")
                
                # Apply temperature
                next_token_logits = next_token_logits / max(temperature, 0.1)
                
                # Apply top-k filter
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.shape[-1]))
                    next_token_logits = torch.full_like(next_token_logits, -float('Inf'))
                    next_token_logits[top_k_indices] = top_k_values
                
                # Apply top-p (nucleus) filter
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumsum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumsum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Convert to probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Select token
                if use_argmax:
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                
                token_value = next_token.item()
                generated_tokens.append(token_value)
                
                # Append to input
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Stop if EOS or padding
                if token_value == self.tokenizer.eos_token_id or token_value == self.tokenizer.pad_token_id:
                    print(f"   🛑 Stopped at step {i+1} (EOS/PAD token)")
                    break
        
        # Decode
        if not generated_tokens:
            return "[No tokens generated - model may need more training]"
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip() if generated_text else "[Empty output]"
    
    def chat(self, user_input, max_length=150, temperature=0.7):
        """
        Chat mode - nhập câu hỏi và nhận câu trả lời
        
        Args:
            user_input: str - câu hỏi
            max_length: int - độ dài max trả lời
            temperature: float - độ sáng tạo
        
        Returns:
            str - câu trả lời
        """
        # Format như training
        prompt = f"Câu hỏi: {user_input}\nTrả lời:"
        
        response = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return response
    
    def interactive_chat(self, max_length=150, temperature=0.7):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("💬 MINI CHAT GPT - Interactive Mode")
        print("="*60)
        print("Nhập câu hỏi (gõ 'quit' hoặc 'exit' để thoát)\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = self.chat(user_input, max_length, temperature)
                print(f"🤖 Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break


# ============================================================================
# 3. MAIN - DEMO USAGE
# ============================================================================

def main():
    # ===== CONFIG =====
    checkpoint_path = r"D:\chuyen_nganh\ChatGPT\outputs\vilqa_chatgpt_v2\best_model.pt"  # ← Thay đổi đường dẫn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===== LOAD MODEL =====
    print("\n" + "="*60)
    print("🚀 Loading Mini Chat GPT Model")
    print("="*60 + "\n")
    
    inference = ChatGPTInference(checkpoint_path, device)
    
    # ===== TEST EXAMPLES =====
    print("="*60)
    print("✅ Model loaded successfully!")
    print("="*60)
    
    test_prompts = [
        "Câu hỏi: Hôm nay là ngày gì?\nTrả lời:",
        "Câu hỏi: 1 + 1 = bao nhiêu?\nTrả lời:",
        "Câu hỏi: Việt Nam có bao nhiêu tỉnh?\nTrả lời:"
    ]
    
    print("\n📝 Test Examples:\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{i}. Prompt: {prompt}")
        response = inference.generate(prompt, max_length=100, temperature=0.7)
        print(f"   Response: {response}")
        print()
    
    # ===== CHAT MODE =====
    print("\n" + "="*60)
    print("🎯 Starting Interactive Chat Mode...")
    print("="*60)
    
    inference.interactive_chat(max_length=150, temperature=0.7)

if __name__ == "__main__":
    main()