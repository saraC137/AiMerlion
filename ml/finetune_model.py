"""
LoRA Fine-Tuning Script for Resume Extraction
Stable version for RTX 3060 (12GB VRAM)
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

import json
import jsonlines
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import os
from datetime import datetime
import config

class ResumeFineTuner:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct", output_dir="resume_model_finetuned"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("="*70)
        print("RESUME EXTRACTION MODEL FINE-TUNING")
        print("="*70)
        print(f"Base model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Output directory: {output_dir}")
        
        if self.device == "cpu":
            print("\n⚠️ WARNING: No GPU detected. Training will be VERY slow on CPU.")
    
    def load_data(self, train_file="train_data.jsonl", val_file="val_data.jsonl"):
        print("\n" + "-"*70)
        print("LOADING DATA")
        print("-"*70)
        
        train_data, val_data = [], []
        with jsonlines.open(train_file) as reader:
            for item in reader:
                train_data.append(self._format_training_example(item))
        with jsonlines.open(val_file) as reader:
            for item in reader:
                val_data.append(self._format_training_example(item))
        
        print(f"Success: Training examples: {len(train_data)}")
        print(f"Success: Validation examples: {len(val_data)}")
        
        return Dataset.from_list(train_data), Dataset.from_list(val_data)
    
    def _format_training_example(self, item):
        resume_text = item['input']
        labels = item['output'][0]
        name = labels.get('candidate_name', '')
        email = labels.get('contact_info', {}).get('email', [])
        phone = labels.get('contact_info', {}).get('phone', [])
        dob = labels.get('contact_info', {}).get('date_of_birth', '')
        email = email[0] if email else ''
        phone = phone[0] if phone else ''
        
        instruction = """Extract the following information from this resume and return as JSON:
- name
- email
- phone
- date_of_birth (YYYY-MM-DD)

Resume text:
{resume_text}

Return valid JSON only:
{{"name": "...", "email": "...", "phone": "...", "date_of_birth": "..."}}"""
        
        output = json.dumps({
            "name": name,
            "email": email,
            "phone": phone,
            "date_of_birth": dob
        }, ensure_ascii=False)
        
        text = f"<|im_start|>user\n{instruction.format(resume_text=resume_text[:3000])}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        return {"text": text}
    
    def setup_model(self):
        print("\n" + "-"*70)
        print("SETTING UP MODEL")
        print("-"*70)
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model (optimized 4-bit mode)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=True,            # ✅ More stable than 8-bit for RTX 3060
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model ready! Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    
    def tokenize_data(self, train_dataset, val_dataset):
        print("\n" + "-"*70)
        print("TOKENIZING DATA")
        print("-"*70)
        
        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=2048, padding="max_length")
        
        return (
            train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"]),
            val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        )
    
    def train(self, train_dataset, val_dataset, num_epochs=3, batch_size=1, learning_rate=2e-4):
        print("\n" + "-"*70)
        print("STARTING TRAINING")
        print("-"*70)
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none",
            warmup_steps=50,
            optim="adamw_torch",
            gradient_checkpointing=False  # ✅ Disabled for stability
        )
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        print("\nInfo: Training started... (2–4 hours)")
        print("-"*70)
        trainer.train()
        print("\n✅ Training complete!")
        return trainer
    
    def save_model(self, trainer):
        print("\n" + "-"*70)
        print("SAVING MODEL")
        print("-"*70)
        
        output_path = f"{self.output_dir}/final_model"
        trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"✅ Model saved to: {output_path}")
        self._create_ollama_modelfile(output_path)
    
    def _create_ollama_modelfile(self, model_path):
        modelfile = f"""FROM {model_path}

TEMPLATE \"\"\"<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\" 

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"You are a precise resume data extraction assistant. Extract only the requested information and return valid JSON.\"\"\""""
        path = f"{self.output_dir}/Modelfile"
        with open(path, "w") as f:
            f.write(modelfile)
        print(f"✅ Ollama Modelfile created at: {path}")
        print(f"Run: ollama create resume-extractor -f {path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a resume extraction model.")
    parser.add_argument("--train_file", type=str, default="train_data.jsonl", help="Path to the training data file.")
    parser.add_argument("--val_file", type=str, default="val_data.jsonl", help="Path to the validation data file.")
    args = parser.parse_args()

    print("="*70)
    print("RESUME EXTRACTION MODEL FINE-TUNING")
    print("="*70)
    
    if not os.path.exists(args.train_file) or not os.path.exists(args.val_file):
        print(f"❌ Missing data files: {args.train_file} or {args.val_file}")
        return
    
    finetuner = ResumeFineTuner()
    train_ds, val_ds = finetuner.load_data(train_file=args.train_file, val_file=args.val_file)
    finetuner.setup_model()
    train_tok, val_tok = finetuner.tokenize_data(train_ds, val_ds)
    
    print("\n⚠️ Training will heavily use GPU memory. Close other GPU apps.")
    start = datetime.now()
    
    trainer = finetuner.train(train_tok, val_tok)
    
    finetuner.save_model(trainer)
    print(f"\n⏱ Training duration: {datetime.now() - start}")
    print("✅ Done! Ready for Ollama deployment.")


if __name__ == "__main__":
    import argparse
    main()
