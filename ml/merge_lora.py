
# Save as merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "resume_model_finetuned/final_model"
)

print("Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
output_path = "resume_model_merged"
merged_model.save_pretrained(output_path, safe_serialization=True)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("resume_model_finetuned/final_model")
tokenizer.save_pretrained(output_path)

print(f"✅ Merged model saved to: {output_path}")
print(f"Size should be ~28 GB")