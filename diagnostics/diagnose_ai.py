
import ollama
import sys

MODEL_NAME = "resume-extractor:latest"

try:
    print(f"--- AI Diagnostic Tool ---")
    print(f"Attempting to connect to Ollama and use model: {MODEL_NAME}")
    
    # Check if the model exists
    try:
        all_models = ollama.list()
        if not any(m['name'] == MODEL_NAME for m in all_models['models']):
            print(f"\n❌ ERROR: The model '{MODEL_NAME}' was not found by Ollama.")
            print(f"Please ensure you have created the model by running this command:")
            print(f"ollama create resume-extractor -f resume_model_finetuned/Modelfile")
            sys.exit(1)
        else:
            print(f"✅ Model '{MODEL_NAME}' found.")

    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to the Ollama server to verify the model list.")
        print(f"Please make sure the Ollama application is running on your computer.")
        print(f"Detailed error: {e}")
        sys.exit(1)

    # Attempt to use the model
    print("\nAttempting to get a response from the model...")
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    
    print("\n✅ SUCCESS: Successfully connected to Ollama and got a response.")
    print(f"Response from model: {response['message']['content']}")

except Exception as e:
    print(f"\n❌ FATAL ERROR: The AI model failed to respond.")
    print(f"This is the reason the AI is not being used in the main application.")
    print(f"The error occurs when calling `ollama.chat()`.")
    print(f"Please ensure the Ollama application is running and is not blocked by a firewall.")
    print(f"\nDetailed error: {e}")
