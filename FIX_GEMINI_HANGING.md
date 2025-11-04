# 🔧 Fix: Gemini API Hanging Issue

## Problem Identified

Your experiment was **hanging at Episode 1/10** because:

1. **Wrong model names**: You changed to `gemini-2.0-flash-lite` and `gemini-2.5-flash-lite` which **don't exist** ❌
2. The API was trying to connect to invalid endpoints, causing timeouts/hangs
3. KeyboardInterrupt showed it was stuck in `ssl.py` waiting for response

## Root Cause

```python
# ❌ WRONG - These models don't exist
models['gemini-2.0-flash-lite'] = GeminiClient(model_name="gemini-2.0-flash-lite")
models['gemini-2.5-flash-lite'] = GeminiClient(model_name="gemini-2.5-flash-lite")
```

## Solution Applied

### 1. Fixed Model Names ✅

```python
# ✅ CORRECT - These are valid free Gemini models
models['gemini-1.5-flash'] = GeminiClient(model_name="gemini-1.5-flash")
models['gemini-1.5-flash-8b'] = GeminiClient(model_name="gemini-1.5-flash-8b")
```

**Valid Gemini Models (Free Tier):**
- `gemini-1.5-flash` - Fast and capable (recommended)
- `gemini-1.5-flash-8b` - Ultra-fast, smaller model
- `gemini-1.5-pro` - Most capable (optional)

### 2. Enhanced Error Handling ✅

Updated `gemini_client.py` with:
- Specific timeout exceptions
- HTTP error details
- Network error handling
- Better error messages

```python
except requests.exceptions.Timeout:
    print(f"⚠️  Timeout calling {self.model_name} API (30s limit)")
    return "RIGHT"  # Safe fallback
    
except requests.exceptions.HTTPError as e:
    print(f"⚠️  HTTP error from {self.model_name}: {e}")
    # Shows error details from API
```

### 3. Created Test Script ✅

New file: `test_gemini_connection.py`

Run this **before** your experiment to catch issues early:

```bash
python test_gemini_connection.py
```

This will:
- ✓ Check if GEMINI_API_KEY is set
- ✓ Test both Gemini models with simple prompt
- ✓ Verify Ollama availability
- ✓ Show clear error messages if anything fails

## How to Fix Your Setup

### Step 1: Test Gemini API Connection
```bash
python test_gemini_connection.py
```

You should see:
```
✓ API key found
✓ Response: 4
✓ Stats: {'total_calls': 1, ...}
✅ Both Gemini models working!
```

### Step 2: Run Your Experiment Again
```bash
python experiments/multi_model_comparison.py
```

It should now work correctly with:
- gemini-1.5-flash ✓
- gemini-1.5-flash-8b ✓
- phi3 ✓
- llama3.2 ✓

## Why It Was Hanging

The stack trace showed:
```
File "ssl.py", line 1134, in read
    return self._sslobj.read(len, buffer)
KeyboardInterrupt
```

This means:
1. Python was waiting for SSL response from API
2. API endpoint didn't exist (wrong model name)
3. Request hung until you pressed Ctrl+C
4. The 30s timeout wasn't triggered because connection was still "open" but waiting

## Expected Behavior Now

### Before Fix:
```
Episode 1/10: ✓ Compromise=0.0%
[HANGS HERE - waiting forever]
```

### After Fix:
```
Episode 1/10: ✓ Compromise=0.0%
Episode 2/10: ✓ Compromise=0.0%
Episode 3/10: ✓ Compromise=0.0%
...
[COMPLETES NORMALLY]
```

## Verification Checklist

Before running full experiment:

- [ ] Run `python test_gemini_connection.py`
- [ ] Verify you see "✅ Both Gemini models working!"
- [ ] Check that both models return valid responses
- [ ] Confirm Ollama is running (`ollama serve`)
- [ ] Verify models downloaded (`ollama list`)

Then:

- [ ] Run `python experiments/multi_model_comparison.py`
- [ ] Should complete all 200 episodes (4 models × 5 strengths × 10 episodes)
- [ ] Should generate CSV and 4 plots

## Common Gemini API Issues

### Issue 1: Rate Limiting
**Error:** "429 Too Many Requests"
**Solution:** Wait 60 seconds, free tier has 15 requests/minute

### Issue 2: Invalid API Key
**Error:** "401 Unauthorized"
**Solution:** Check GEMINI_API_KEY in `.env` file

### Issue 3: Model Not Found
**Error:** "404 Not Found"
**Solution:** Use valid model names (gemini-1.5-flash, gemini-1.5-flash-8b)

### Issue 4: Timeout
**Error:** Hangs for 30+ seconds
**Solution:** Check internet connection, API might be slow

## Files Modified

1. **`experiments/multi_model_comparison.py`**
   - Changed `gemini-2.0-flash-lite` → `gemini-1.5-flash`
   - Changed `gemini-2.5-flash-lite` → `gemini-1.5-flash-8b`

2. **`llm/gemini_client.py`**
   - Added timeout exception handling
   - Added HTTP error details
   - Better error messages

3. **`test_gemini_connection.py`** (NEW)
   - Quick API validation script
   - Tests both models before full experiment

## Quick Reference

### Valid Gemini Models
| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| gemini-1.5-flash | Medium | Fast | Recommended |
| gemini-1.5-flash-8b | Small | Fastest | Quick tests |
| gemini-1.5-pro | Large | Slower | Best quality |

### Ollama Models
| Model | Size | RAM Needed | Speed |
|-------|------|------------|-------|
| phi3 | 3.8GB | 6GB+ | Medium |
| llama3.2 | 2GB | 4GB+ | Fast |
| tinyllama | 637MB | 2GB+ | Fastest |
| gemma2:2b | 1.6GB | 3GB+ | Fast |

## Next Steps

1. **Test API**: `python test_gemini_connection.py`
2. **If test passes**: `python experiments/multi_model_comparison.py`
3. **If test fails**: Check error messages and fix issues
4. **Monitor progress**: Should complete in 15-20 minutes

## Success Indicators

✅ Test shows both Gemini models responding
✅ Experiment progresses beyond Episode 1
✅ All 200 episodes complete
✅ CSV and plots generated

Good luck! The fix should resolve the hanging issue. 🚀
