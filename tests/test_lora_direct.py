"""
test_lora_direct.py

💅✨ FAIRY CODEMOTHER'S DIRECT LORA TESTER ✨💅

Tests your fine-tuned LoRA model DIRECTLY (before GGUF export)
to determine if:
  A) Training worked but GGUF export broke the weights
  B) Training itself didn't work

If this script produces GOOD JSON output, the problem is in GGUF export.
If this script produces GIBBERISH, the problem is in training itself.

Usage:
    python test_lora_direct.py
    python test_lora_direct.py --lora-path resume_model_stage1\lora_model
    python test_lora_direct.py --test-id 2
"""

import argparse
import json
import os
import sys
import time

os.environ["XFORMERS_DISABLED"] = "1"

TEST_SAMPLES = [
    {
        "id": 1,
        "name": "Easy - SG Chinese Name",
        "prompt": "Extract structured data from this resume. Return valid JSON with these fields where available: name, email, phone, date_of_birth, location, summary, hard_skills (array), soft_skills (array), experience (array of {title, company, duration}), education (array of {degree, institution}), certifications (array), languages (array), function (job function category), industry (industry category).\n\nResume:\nLim Kai Wen\nEmail: kaiwen.lim@gmail.com\nPhone: +65 9876 5432\nDate of Birth: 10 May 1995\n\nAccountant\nKPMG Singapore Pte Ltd\nMarch 2020 - Present\n- Prepared financial statements for SME clients\n- Conducted GST audits and IRAS submissions\n\nAudit Associate\nErnst & Young LLP Singapore\nJuly 2017 - February 2020\n- Performed statutory audits for listed companies\n\nBachelor of Accountancy\nSingapore Management University (SMU)\n2013 - 2017 | GPA: 3.6 / 4.0\n\nSkills: SAP FICO, Microsoft Excel, Xero, QuickBooks, IFRS, SQL\nCertifications: CPA Singapore (ISCA), ACCA Affiliate\nLanguages: English (Fluent), Mandarin (Native)",
        "expected_name": "Lim Kai Wen",
    },
    {
        "id": 2,
        "name": "Medium - MY binti Name",
        "prompt": "Extract structured data from this resume. Return valid JSON with these fields where available: name, email, phone, date_of_birth, location, summary, hard_skills (array), soft_skills (array), experience (array of {title, company, duration}), education (array of {degree, institution}), certifications (array), languages (array), function (job function category), industry (industry category).\n\nResume:\nSiti Nurhaliza binti Mohd Yusoff\nNo. Telefon: +60 13-456 7890\nEmel: siti.nurhaliza90@gmail.com\nTarikh Lahir: 25 Disember 1990\n\nPengurus Akaun / Account Manager\nMaybank Berhad | Kuala Lumpur\nJanuari 2018 - Kini\n- Menguruskan portfolio 150 pelanggan korporat\n\nEksekutif Pemasaran / Marketing Executive\nAirAsia Berhad | Sepang, Selangor\nJun 2014 - Disember 2017\n- Merancang kempen pemasaran digital\n\nPendidikan:\nSarjana Muda Pentadbiran Perniagaan\nUniversiti Kebangsaan Malaysia (UKM)\n2010 - 2014 | CGPA: 3.38 / 4.00\n\nKemahiran: Salesforce, Google Analytics, Hootsuite, Canva, Microsoft Office\nBahasa: Bahasa Malaysia (Ibunda), English (Fasih), Mandarin (Asas)",
        "expected_name": "Siti Nurhaliza binti Mohd Yusoff",
    },
    {
        "id": 3,
        "name": "Hard - Tab-separated D/O",
        "prompt": "Extract structured data from this resume. Return valid JSON with these fields where available: name, email, phone, date_of_birth, location, summary, hard_skills (array), soft_skills (array), experience (array of {title, company, duration}), education (array of {degree, institution}), certifications (array), languages (array), function (job function category), industry (industry category).\n\nResume:\nPERSONAL PARTICULARS\nName\t:\tKavitha D/O Ramasamy\nNRIC\t:\tS8512345C\nD.O.B\t:\t15/07/1985\nH/P\t:\t92345678\nEmail\t:\tkavitha.rama@yahoo.com.sg\n\nPrincipal : Temasek Primary School (MOE)\nDuration : Aug 2015 - Present\n- Lead Science and Math department\n\nTeacher : Jurong West Primary School (MOE)\nDuration : Jan 2009 - Jul 2015\n- Taught English and Mathematics\n\nPGDE (Primary)\nNational Institute of Education (NIE), NTU\n2008 - 2009\n\nBA (English Literature)\nNational University of Singapore (NUS)\n2004 - 2008\n\nSkills: Cambridge Assessment, SLS Platform, Google Classroom\nLanguages: English, Tamil, Malay",
        "expected_name": "Kavitha D/O Ramasamy",
    },
]


def test_lora(lora_path, test_id=None):
    print()
    print("=" * 60)
    print("  DIRECT LORA MODEL TESTER")
    print("=" * 60)
    print(f"  LoRA path: {lora_path}")
    print()

    if not os.path.exists(lora_path):
        print(f"  ERROR: LoRA path not found: {lora_path}")
        return

    config_path = os.path.join(lora_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"  Base model: {config.get('base_model_name_or_path', '?')}")
        print(f"  LoRA rank:  {config.get('r', '?')}")
        print(f"  LoRA alpha: {config.get('lora_alpha', '?')}")

    print()
    print("  Loading model (30-60 seconds)...")

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    # Blackwell fast_generate bypass
    if hasattr(model, "base_model") and hasattr(model.base_model, "_old_generate"):
        model.base_model.generate = model.base_model._old_generate
        print("  Applied Blackwell fast_generate bypass")

    if test_id:
        samples = [s for s in TEST_SAMPLES if s["id"] == test_id]
    else:
        samples = TEST_SAMPLES

    results = []
    for sample in samples:
        print()
        print("-" * 60)
        print(f"  TEST {sample['id']}: {sample['name']}")
        print("-" * 60)

        messages = [{"role": "user", "content": sample["prompt"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(model.device)

        print(f"  Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"  Generating (1-3 minutes)...")

        start = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - start
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"  Time: {elapsed:.1f}s | Tokens: {len(gen_ids)}")
        print()

        is_json = False
        parsed = None
        try:
            parsed = json.loads(gen_text)
            is_json = True
        except (json.JSONDecodeError, ValueError):
            pass

        non_ascii = sum(1 for c in gen_text if ord(c) > 127)
        is_gibberish = (non_ascii / max(len(gen_text), 1)) > 0.3

        if is_json:
            print("  RESULT: Valid JSON!")
            print(f"  {json.dumps(parsed, indent=2, ensure_ascii=False)[:500]}")
            name = parsed.get("name", "")
            if name == sample["expected_name"]:
                print(f"\n  Name: CORRECT ({name})")
            else:
                print(f"\n  Name: got '{name}', expected '{sample['expected_name']}'")
        elif is_gibberish:
            print("  RESULT: GIBBERISH")
            print(f"  Preview: {gen_text[:200]}")
        else:
            print("  RESULT: Not valid JSON")
            print(f"  Preview: {gen_text[:300]}")

        results.append({
            "test_id": sample["id"],
            "valid_json": is_json,
            "gibberish": is_gibberish,
            "time": elapsed,
        })

    # Summary
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    valid = sum(1 for r in results if r["valid_json"])
    gib = sum(1 for r in results if r["gibberish"])
    print(f"  Valid JSON:  {valid}/{len(results)}")
    print(f"  Gibberish:   {gib}/{len(results)}")
    print()

    if valid == len(results):
        print("  VERDICT: LoRA works! Problem is GGUF export.")
        print("  FIX: Try --gguf f16 instead of q8_0")
    elif gib == len(results):
        print("  VERDICT: LoRA itself is broken. Training issue.")
        print("  FIX: Check learning rate, epochs, data format")
    else:
        print("  VERDICT: Partially working. Needs investigation.")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, default="resume_model_stage1/lora_model")
    parser.add_argument("--test-id", type=int, default=None)
    args = parser.parse_args()
    test_lora(args.lora_path, args.test_id)