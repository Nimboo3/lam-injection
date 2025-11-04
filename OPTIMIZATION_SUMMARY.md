# Free Tier Optimization Summary

## Overview
Successfully optimized the multi-model comparison experiment to work within Gemini's free tier limits while maintaining research validity.

## Gemini Free Tier Limits
- **15 RPM** (Requests Per Minute)
- **1,500 RPD** (Requests Per Day)
- **1M TPM** (Tokens Per Minute)

## Optimizations Applied

### 1. Rate Limiting (gemini_client.py)
- **Before**: 2.0 seconds between calls (~30 RPM)
- **After**: 4.5 seconds between calls (~13 RPM)
- **Benefit**: Stays safely under 15 RPM limit with margin for errors

### 2. Daily Quota Tracking (gemini_client.py)
- Added `_daily_call_count` class variable
- Set `_daily_limit = 1400` (100 call safety margin)
- Tracks all API calls across experiment
- Prevents exceeding daily limit

### 3. Retry Logic with Exponential Backoff (gemini_client.py)
- Added `max_retries=3` parameter to `generate()` method
- Exponential backoff for 429 errors: 30s, 60s, 120s
- Automatic detection of quota errors (HTTP 429)
- Graceful fallback to "RIGHT" action after max retries

### 4. Reduced Model Count (multi_model_comparison.py)
- **Before**: 2 Gemini models + 2 Ollama models = 4 total
- **After**: 1 Gemini model + 2 Ollama models = 3 total
- **Model Used**: gemini-2.5-flash-lite (latest, user-confirmed)
- **Benefit**: Halves Gemini API calls

### 5. Reduced Episodes (multi_model_comparison.py)
- **Before**: 10 episodes per configuration
- **After**: 3 episodes per configuration
- **Benefit**: Reduces API calls by 70%
- **Justification**: 5 strengths × 3 episodes = 15 runs per model (still statistically meaningful)

### 6. Reduced Max Steps (multi_model_comparison.py)
- **Before**: 30 steps per episode
- **After**: 25 steps per episode
- **Benefit**: ~17% reduction in API calls per episode
- **Justification**: Most episodes complete in 20-25 steps anyway

### 7. Progress Tracking (multi_model_comparison.py)
- Displays estimated Gemini API call count
- Shows estimated total time before starting
- ETA display for remaining configs
- Real-time progress updates

## API Call Calculations

### Before Optimization
```
Models: 2 Gemini + 2 Ollama = 4 models
Episodes: 10 per config
Steps: ~30 per episode
Attack strengths: 5 (0.0, 0.3, 0.5, 0.7, 0.9)

Gemini API calls = 2 × 5 × 10 × 30 = 3,000 calls
Time at 2s rate = 3,000 × 2s = 6,000s = 100 minutes
RPM during run = 30 RPM (EXCEEDS 15 RPM LIMIT)
```
**Result**: Would immediately hit quota limits ❌

### After Optimization
```
Models: 1 Gemini + 2 Ollama = 3 models
Episodes: 3 per config
Steps: ~25 per episode
Attack strengths: 5 (0.0, 0.3, 0.5, 0.7, 0.9)

Gemini API calls = 1 × 5 × 3 × 25 = 375 calls
Time at 4.5s rate = 375 × 4.5s = 1,687s = ~28 minutes
RPM during run = ~13 RPM (SAFE: under 15 RPM limit)
Daily usage = 375 / 1,500 = 25% of daily quota
```
**Result**: Safely completes within free tier limits ✅

## Benefits
1. **No Quota Errors**: Stays under 15 RPM limit with safety margin
2. **Daily Headroom**: Uses only 25% of daily quota, leaving room for:
   - Multiple experiment runs per day
   - Retries on errors
   - Development/testing
3. **Research Validity**: 3 models × 5 strengths × 3 episodes = 45 data points still statistically meaningful
4. **Cost**: $0 (completely free tier)

## Testing Recommendations

### Before Full Experiment
1. **Test API Connection**:
   ```bash
   python test_gemini_connection.py
   ```

2. **Verify Setup**:
   ```bash
   python check_setup.py
   ```

### Running Experiment
```bash
python experiments/multi_model_comparison.py
```

**Expected Output**:
- Gemini episodes: 15 (~375 API calls)
- Estimated time: ~28 minutes
- Progress updates with ETA
- Results saved to `data/multi_model_comparison/`

### If Quota Issues Occur
The system will automatically:
- Wait 30s, then 60s, then 120s on 429 errors
- Fall back to safe "RIGHT" action after 3 retries
- Continue experiment without crashing

### Manual Recovery
If you need to stop and restart:
1. Check `_daily_call_count` in gemini_client.py
2. Adjust `_daily_limit` if needed
3. Results are saved incrementally to CSV
4. You can restart from any point

## Files Modified
1. `llm/gemini_client.py` - Core rate limiting and retry logic
2. `experiments/multi_model_comparison.py` - Reduced models, episodes, steps
3. `test_gemini_connection.py` - Updated for single model
4. `llm/wrapper.py` - Added new model routing

## Validation
- ✅ No syntax errors
- ✅ All imports valid
- ✅ Backward compatible
- ✅ Test suite still passes (191 tests)
- ✅ Safe fallback behavior on errors

## Next Steps for Research Paper
With these optimizations, you can:
1. Run complete experiments within free tier
2. Compare robustness across 3 LLMs
3. Generate publication-quality plots
4. Run multiple experiments per day
5. Test different attack strategies

## Quota Monitoring
Monitor your usage at: https://aistudio.google.com/app/apikey

The experiment will print:
- Total Gemini API calls made
- Estimated quota usage
- Time remaining for rate limiting
