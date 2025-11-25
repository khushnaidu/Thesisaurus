# Troubleshooting Guide

## Common Dependency Errors

### Error 1: NumPy Version Incompatibility

**Error Message:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```

**Cause:** NumPy 2.x incompatible with older scipy/scikit-learn versions

### Error 2: huggingface_hub ImportError

**Error Message:**
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**Cause:** Old sentence-transformers incompatible with new huggingface_hub

---

## ✅ **Universal Solution (Recommended)**

This fixes all dependency issues:

```bash
cd phase1_data_preparation

# Run the fix script
bash fix_dependencies.sh
```

This will:
1. Uninstall conflicting packages
2. Install compatible versions from requirements.txt
3. Verify installation

### ✅ **Manual Fix (Alternative)**

If the script doesn't work:

```bash
# Activate your environment
conda activate phase1  # or: source ../venv/bin/activate

# Clean install
pip uninstall -y sentence-transformers transformers huggingface-hub numpy
pip install --no-cache-dir -r requirements.txt

# Verify
python -c "from sentence_transformers import SentenceTransformer; print('✓ Works!')"
```

### ✅ **Solution 3: Database Demo Only (No Fix Needed)**

If you just want to test the database functionality without fixing NumPy:

```bash
cd scripts
python demo_database_only.py
```

This runs a simplified demo that doesn't require sentence-transformers or FAISS.

---

## Testing Options

### Option A: Full Demo (Requires NumPy Fix)
```bash
cd scripts
python demo_queries.py
```

Shows:
- ✓ Database queries
- ✓ Semantic search
- ✓ Combined queries

### Option B: Database Only (No Dependencies)
```bash
cd scripts
python demo_database_only.py
```

Shows:
- ✓ Database queries
- ✗ Semantic search (skipped)

### Option C: Verify Setup Only
```bash
cd phase1_data_preparation
python verify_setup.py
```

Shows:
- ✓ File structure
- ✓ Data integrity
- ✓ No imports needed

---

## Environment Recommendations

### Using Conda (Your Current Setup)

You're using Anaconda's base environment. For better isolation:

```bash
# Create a dedicated conda environment
conda create -n phase1 python=3.10
conda activate phase1

# Install dependencies
cd phase1_data_preparation
pip install -r requirements.txt
```

### Using Project Venv

```bash
# Activate the existing venv
cd NLP_Project
source venv/bin/activate

# Fix NumPy
pip install --force-reinstall numpy==1.24.3

# Test
cd phase1_data_preparation/scripts
python demo_queries.py
```

---

## Verification Commands

### Check Your NumPy Version
```bash
python -c "import numpy; print(numpy.__version__)"
```

Should show: `1.24.3` (or similar compatible version)

### Check sentence-transformers
```bash
python -c "from sentence_transformers import SentenceTransformer; print('✓ Works!')"
```

### Check FAISS
```bash
python -c "import faiss; print('✓ Works!')"
```

### Check Database
```bash
sqlite3 outputs/papers.db "SELECT COUNT(*) FROM papers;"
```

Should show: `18`

---

## Quick Diagnostic

Run this to see what's installed:

```bash
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except: print('NumPy: NOT INSTALLED')
try:
    import sentence_transformers
    print(f'sentence-transformers: ✓')
except Exception as e: print(f'sentence-transformers: ✗ ({e})')
try:
    import faiss
    print(f'FAISS: ✓')
except: print('FAISS: ✗')
"
```

---

## Still Having Issues?

1. **Try a fresh environment:**
   ```bash
   conda create -n phase1_test python=3.10
   conda activate phase1_test
   cd phase1_data_preparation
   pip install -r requirements.txt
   cd scripts
   python demo_queries.py
   ```

2. **Or use database-only demo:**
   ```bash
   python demo_database_only.py
   ```

3. **Check the outputs are valid:**
   ```bash
   python verify_setup.py
   ```

All the outputs (CSV, database, vector index) are pre-generated, so you can verify they exist without running any scripts that require external dependencies.

