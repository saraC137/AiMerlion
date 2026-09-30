from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="resume_model_finetuned/lora_model"
)

model.save_pretrained_gguf(
    "resume_model_finetuned/gguf_model",
    tokenizer,
    quantization_method="q4_k_m"
)

print("✅ GGUF export done!")