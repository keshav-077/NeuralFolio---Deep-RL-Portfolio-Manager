# 🚀 Deploy NeuralFolio to Hugging Face Spaces

## Quick Deploy (5 minutes)

### Step 1: Create Your Space

1. Go to: **https://huggingface.co/new-space**
2. Sign in with your account (**keshav775**)
3. Fill in:
   - **Space name**: `neuralfolio` (or any name you want)
   - **License**: MIT
   - **SDK**: **Gradio** ⚠️ (Very important!)
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public
4. Click **"Create Space"**

### Step 2: Clone Your New Space

```bash
git clone https://huggingface.co/spaces/keshav775/neuralfolio
cd neuralfolio
```

### Step 3: Copy Files from Your GitHub

From your GitHub repo, copy these files to the Space folder:

**Required files:**
```bash
# Copy from your repo to the Space folder:
- app.py
- environment.py
- custom_policy.py
- fetch_market_data.py
- llm_analysis_rag.py
- evaluate_baselines.py
- requirements.txt
- README_SPACE.md (rename to README.md)
- .gitattributes
- checkpoints/ (folder with .gitkeep)
- data/ (folder with .gitkeep)
```

**Easy way:** Just copy everything from your repo to the Space folder.

### Step 4: Rename Space README

```bash
# In your Space folder:
mv README_SPACE.md README.md
```

### Step 5: Push to Hugging Face

```bash
git add .
git commit -m "Deploy NeuralFolio Gradio app"
git push
```

### Step 6: Wait for Build

1. Go to: `https://huggingface.co/spaces/keshav775/neuralfolio`
2. Click **"Building"** tab to watch progress
3. First build takes **5-10 minutes** (installing PyTorch, etc.)
4. When done, your app will be **LIVE!** 🎉

---

## Expected Behavior

Since you don't have a trained model checkpoint yet, the app will:
- ✅ Show the Historical Data Analyst tab (works without model)
- ✅ Fetch and display market data
- ⚠️ Show warnings for tabs that need the trained model

### To Get Full Functionality

Train a model locally and add it to your Space:

```bash
# In your local project:
python scripts/fetch_market_data.py --start 2015-01-01 --end 2020-12-31 --filename data/train_data.csv
python scripts/train.py --agent sac --timesteps 20000

# Then copy checkpoints/sac_portfolio_model.zip to your Space:
cp checkpoints/sac_portfolio_model.zip /path/to/neuralfolio/checkpoints/
cd /path/to/neuralfolio
git lfs track "checkpoints/*.zip"
git add checkpoints/
git commit -m "Add trained SAC model"
git push
```

---

## Troubleshooting

### Build Failed?
- Check the **Build logs** tab on your Space
- Common issue: Wrong SDK (must be "Gradio", not "Streamlit")

### Out of Memory?
- The free CPU tier might struggle with the 3B LLM
- Edit `llm_analysis_rag.py` line ~24: use `Qwen/Qwen2.5-1.5B-Instruct`
- Or upgrade to **T4 GPU** ($0.60/hour) in Space settings

### Slow LLM?
- Normal on CPU! LLM takes 30-60s per analysis
- Add a "⏳ Please wait..." message in the app
- Or upgrade to GPU hardware

---

## Your Live URLs

- **GitHub**: https://github.com/keshav-077/NeuralFolio---Deep-RL-Portfolio-Manager
- **HF Space** (after deploy): https://huggingface.co/spaces/keshav775/neuralfolio

---

## Alternative: Direct GitHub → HF Space

You can also link your GitHub repo directly:

1. Create Space as above
2. In Space settings, enable "**Link to GitHub repo**"
3. Connect: `keshav-077/NeuralFolio---Deep-RL-Portfolio-Manager`
4. Auto-deploys on every GitHub push!

---

Need help? Ask in HF Discussions or tag me!

**Created by**: keshav-077  
**Project**: NeuralFolio - Deep RL Portfolio Manager
