"""
🔥 NUCLEAR DIAGNOSTIC - Find EXACTLY which file Python is using!
Run this from your project directory!
"""

import os
import sys
import importlib

print("=" * 70)
print("🔥 NUCLEAR DIAGNOSTIC - Finding the REAL ai_extractor.py")
print("=" * 70)

# Step 1: Show Python path
print("\n📍 Python executable:", sys.executable)
print("📍 Working directory:", os.getcwd())

# Step 2: Find ALL ai_extractor.py files on the system
print("\n🔍 Searching for ALL ai_extractor.py files...")

search_paths = [
    os.getcwd(),
    os.path.dirname(os.getcwd()),
    os.path.expanduser("~"),
]

# Also search in Python path
search_paths.extend(sys.path)

found_files = set()
for search_path in search_paths:
    if not os.path.exists(search_path):
        continue
    for root, dirs, files in os.walk(search_path):
        # Skip virtual environments and common non-project folders
        skip_dirs = ['venv', '.venv', 'env', '.env', 'node_modules', '.git', 'site-packages']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        if 'ai_extractor.py' in files:
            found_files.add(os.path.join(root, 'ai_extractor.py'))
        # Also look for .pyc files
        for f in files:
            if 'ai_extractor' in f and f.endswith('.pyc'):
                found_files.add(os.path.join(root, f))

print(f"\n📄 Found {len(found_files)} ai_extractor files:")
for f in sorted(found_files):
    print(f"   → {f}")

# Step 3: Try to import ai_extractor and see where it comes from
print("\n🔬 Attempting to import ai_extractor...")
try:
    # Clear any cached import first
    if 'ai_extractor' in sys.modules:
        del sys.modules['ai_extractor']
    
    import ai_extractor
    actual_file = ai_extractor.__file__
    print(f"✅ Imported from: {actual_file}")
    
    # Check if this file has the fix
    with open(actual_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if 'using pattern {pattern_used}' in content:
        print("✅ This file HAS the fix!")
    else:
        print("❌ This file does NOT have the fix!")
        print("   This is the problem! Python is importing an OLD version!")
        
except ImportError as e:
    print(f"❌ Could not import ai_extractor: {e}")

# Step 4: Check for __pycache__ directories
print("\n🗑️ Checking for __pycache__ directories...")
pycache_count = 0
for root, dirs, files in os.walk(os.getcwd()):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        pycache_count += 1
        print(f"   Found: {pycache_path}")
        # List contents
        try:
            for f in os.listdir(pycache_path):
                if 'ai_extractor' in f:
                    full_path = os.path.join(pycache_path, f)
                    print(f"      ⚠️ CACHED: {f} (modified: {os.path.getmtime(full_path)})")
        except:
            pass

if pycache_count == 0:
    print("   ✅ No __pycache__ directories found!")

print("\n" + "=" * 70)
print("💡 SOLUTION:")
print("=" * 70)
print("""
If Python is importing from a DIFFERENT location:
1. Delete ALL other ai_extractor.py files
2. Delete ALL __pycache__ folders:
   
   Windows (run in PowerShell):
   Get-ChildItem -Path . -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force
   
   Linux/Mac:
   find . -type d -name "__pycache__" -exec rm -rf {} +

3. Restart your Python/terminal completely
4. Run your script again
""")