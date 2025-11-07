from torch.utils.data import Dataset

class ViLQADataset(Dataset):
    """Dataset cho ViLQA"""
    
    def __init__(self, data, tokenizer, max_length=512, split="train"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"   Loading {split} split: {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = str(item.get("Question", "") or "")
        answer = str(item.get("Answer", "") or "")
        
        # Handle answer format
        if isinstance(answer, dict):
            answer = answer.get("text", [""])[0] if answer.get("text") else ""
        elif isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        question = question.strip()
        answer = answer.strip()

        # Fallback
        if not question:
            question = "Xin hỏi?"
        if not answer:
            answer = "Tôi không biết."
            
        # Format text - QUAN TRỌNG: không thêm </s> để model tự học sinh nó
        text = f"Câu hỏi: {question}\nTrả lời: {answer}"
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        # labels và inputs_id y chang nhau, có điều chỗ nào có padding thì chỗ đó giá trị = -100 = hết.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }