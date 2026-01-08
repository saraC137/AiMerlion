"""
🧹 FAIRY CODEMOTHER'S CACHE BUSTER & VERIFIER
Run this BEFORE running your extractor!

Usage: python fix_cache.py
"""

import os
import shutil
import sys

def find_and_clear_pycache():
    """Find and delete all __pycache__ directories"""
    print("=" * 60)
    print("🧹 CACHE BUSTER - Clearing Python cache files!")
    print("=" * 60)
    
    current_dir = os.getcwd()
    print(f"\n📂 Working directory: {current_dir}")
    
    # Find all __pycache__ directories
    pycache_dirs = []
    pyc_files = []
    
    for root, dirs, files in os.walk(current_dir):
        # Find __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            pycache_dirs.append(pycache_path)
        
        # Find .pyc files
        for f in files:
            if f.endswith('.pyc'):
                pyc_files.append(os.path.join(root, f))
    
    print(f"\n🔍 Found {len(pycache_dirs)} __pycache__ directories")
    print(f"🔍 Found {len(pyc_files)} .pyc files")
    
    # Delete them!
    for pycache in pycache_dirs:
        try:
            shutil.rmtree(pycache)
            print(f"   🗑️ Deleted: {pycache}")
        except Exception as e:
            print(f"   ⚠️ Could not delete {pycache}: {e}")
    
    for pyc in pyc_files:
        try:
            os.remove(pyc)
            print(f"   🗑️ Deleted: {pyc}")
        except Exception as e:
            print(f"   ⚠️ Could not delete {pyc}: {e}")
    
    print("\n✅ Cache cleared!")


def verify_ai_extractor():
    """Check if ai_extractor.py has the fix"""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING ai_extractor.py")
    print("=" * 60)
    
    # Find all ai_extractor.py files
    current_dir = os.getcwd()
    found_files = []
    
    for root, dirs, files in os.walk(current_dir):
        if 'ai_extractor.py' in files:
            found_files.append(os.path.join(root, 'ai_extractor.py'))
    
    print(f"\n📄 Found {len(found_files)} ai_extractor.py file(s):")
    
    for filepath in found_files:
        print(f"\n{'─' * 50}")
        print(f"📂 {filepath}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for NEW fix markers
        has_fix = False
        fix_indicators = [
            "using pattern {pattern_used}",
            "FAIRY CODEMOTHER'S ULTIMATE FIX",
            "pattern_used = i + 1",
            "captured_len < 500",
        ]
        
        for indicator in fix_indicators:
            if indicator in content:
                has_fix = True
                print(f"   ✅ Found fix marker: '{indicator[:40]}...'")
        
        # Check for OLD broken pattern
        old_patterns = [
            '(.*?)(?=EDUCATION|SKILLS|CERTIFICATIONS|AWARDS|$)',
        ]
        
        has_old = False
        for old in old_patterns:
            if old in content:
                has_old = True
                print(f"   ❌ STILL HAS OLD PATTERN: '{old[:50]}...'")
        
        # Find the actual exp_patterns line
        if 'exp_patterns = [' in content:
            # Extract the patterns section
            start = content.find('# 🔍 STEP 2: FIND EXPERIENCE SECTION')
            if start == -1:
                start = content.find('exp_patterns = [')
            
            if start != -1:
                end = content.find('exp_text = ""', start)
                if end != -1:
                    section = content[start:end]
                    print(f"\n   📋 Experience patterns section ({len(section)} chars):")
                    # Show first few lines
                    lines = section.split('\n')[:10]
                    for line in lines:
                        print(f"      {line[:70]}{'...' if len(line) > 70 else ''}")
        
        if has_fix and not has_old:
            print(f"\n   🎉 THIS FILE HAS THE FIX!")
        elif has_old:
            print(f"\n   ❌ THIS FILE STILL HAS THE OLD BROKEN CODE!")
        else:
            print(f"\n   ⚠️ Could not determine fix status")


def main():
    print("🎭 FAIRY CODEMOTHER'S DIAGNOSTIC TOOL 💅")
    print("=" * 60)
    
    # Step 1: Clear cache
    find_and_clear_pycache()
    
    # Step 2: Verify file
    verify_ai_extractor()
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print("""
1. If you see '❌ STILL HAS OLD BROKEN CODE':
   → Your file wasn't properly replaced!
   → Download ai_extractor_FINAL.py again
   → RENAME it to ai_extractor.py (delete the old one first!)

2. If you see '✅ THIS FILE HAS THE FIX':
   → The fix is in place!
   → Run your extractor again - it should work now!

3. Make sure you're running from the SAME directory where
   ai_extractor.py is located!
""")


if __name__ == "__main__":
    main()