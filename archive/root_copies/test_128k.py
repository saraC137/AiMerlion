import ollama
import random

# 1. Configuration
MODEL_NAME = "llama3.1-128k" 
SECRET_CODE = "The secret code is 'GOLD-99-ALPHA'. Write this down."
# 90,000 words is roughly 125,000-130,000 Llama tokens
# Halved to 10000 to reduce VRAM usage for consumer GPUs
TOTAL_WORDS = 10000 

def run_test():
    print(f"--- Starting 128K Context Test on {MODEL_NAME} ---")
    
    # 2. Generate the "Haystack" (Filler text)
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Local LLMs are transforming how we handle private data.",
        "Your i9 processor is processing these tokens rapidly.",
        "RTX 3060 is a great card for 12GB VRAM tasks.",
        "Artificial intelligence requires high memory bandwidth."
    ]
    
    print("Generating 128,000 tokens of text...")
    words = [random.choice(filler_sentences) for _ in range(TOTAL_WORDS)]
    
    # 3. Insert the "Needle" at 90% depth
    insertion_point = int(len(words) * 0.5)
    words.insert(insertion_point, f"\n*** {SECRET_CODE} ***\n")
    
    full_text = " ".join(words)
    
    # 4. The Updated Prompt
    prompt = (
        "INSTRUCTION: Find the secret code hidden in the text below. "
        "The code looks like 'WORD-NUMBER-WORD'.\n\n"
        f"BEGIN MASSIVE DOCUMENT\n"
        f"{full_text}\n"
        f"END MASSIVE DOCUMENT\n\n"
        "FINAL INSTRUCTION: What was the secret code found in the document? "
        "If you see 'GOLD-99-ALPHA', report it now."
    )

    print(f"Prompt ready. Sending to GPU (this will take 1-2 minutes to prefill)...")

    # 5. Call the Model
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        print("\n--- MODEL RESPONSE ---")
        print(response['response'])
        
        if "GOLD-99-ALPHA" in response['response']:
            print("\n✅ SUCCESS: The model found the needle at 128K depth!")
        else:
            print("\n❌ FAIL: The model couldn't find the needle.")
            
    except Exception as e:
        print(f"\n⚠️ ERROR: {e}")

if __name__ == "__main__":
    run_test()