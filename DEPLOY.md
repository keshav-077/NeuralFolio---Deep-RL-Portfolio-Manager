# Deploying NeuralFolio to Hugging Face Spaces

This guide walks you through deploying your NeuralFolio Gradio app to Hugging Face Spaces under your `keshav775` account.

---

## Prerequisites

- A Hugging Face account (`keshav775`)
- Git installed on your machine
- Your NeuralFolio project ready

---

## Step 1: Create a New Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner**: `keshav775`
   - **Space name**: `neuralfolio` (or any name you prefer)
   - **License**: MIT
   - **Select SDK**: **Gradio**
   - **Space hardware**: Start with **CPU basic** (free tier)
   - **Visibility**: Public (or Private if you prefer)
3. Click **Create Space**

---

## Step 2: Prepare Your Repository

Your Space needs these files in the root:

```
neuralfolio/
├── app.py                      # Main Gradio app (rename from scripts/app.py)
├── requirements.txt            # Python dependencies
├── README.md                   # Will show on your Space page
├── checkpoints/                # Pre-trained model weights
│   └── sac_portfolio_model.zip
├── environment.py              # Environment definition (from scripts/)
├── custom_policy.py            # Custom policy networks (from scripts/)
├── llm_analysis_rag.py         # LLM analyst (from scripts/)
├── fetch_market_data.py        # Data fetcher (from scripts/)
└── .gitattributes              # For Git LFS (large model files)
```

### Key Changes Needed

1. **Move `scripts/app.py` to root as `app.py`**:
   ```bash
   cp scripts/app.py app.py
   ```

2. **Update imports in `app.py`**: Change all relative imports from `scripts.` to direct imports:
   ```python
   # Before
   from scripts.environment import PortfolioEnv
   from scripts.llm_analysis_rag import analyze_allocation
   
   # After
   from environment import PortfolioEnv
   from llm_analysis_rag import analyze_allocation
   ```

3. **Copy required modules to root**:
   ```bash
   cp scripts/environment.py .
   cp scripts/custom_policy.py .
   cp scripts/llm_analysis_rag.py .
   cp scripts/fetch_market_data.py .
   ```

4. **Update `requirements.txt`** - ensure it includes all dependencies and specify the smaller LLM for free-tier:
   ```txt
   gradio>=4.0.0
   torch>=2.0.0
   gymnasium>=0.29.0
   stable-baselines3>=2.0.0
   yfinance>=0.2.0
   pandas>=2.0.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   transformers>=4.35.0
   langchain>=0.1.0
   sentence-transformers>=2.2.0
   faiss-cpu>=1.7.0
   accelerate>=0.24.0
   ```

5. **Optimize for free-tier hardware**: In `llm_analysis_rag.py`, ensure you're using the 1.5B model:
   ```python
   MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Not the 3B version
   ```

---

## Step 3: Set Up Git LFS

Your checkpoint files are large (>10MB). Hugging Face Spaces uses Git LFS for large files.

1. **Install Git LFS** (if not installed):
   ```bash
   git lfs install
   ```

2. **Create `.gitattributes`** in your root:
   ```
   *.zip filter=lfs diff=lfs merge=lfs -text
   *.pth filter=lfs diff=lfs merge=lfs -text
   *.bin filter=lfs diff=lfs merge=lfs -text
   ```

---

## Step 4: Push to Hugging Face Spaces

1. **Clone your Space repository**:
   ```bash
   git clone https://huggingface.co/spaces/keshav775/neuralfolio
   cd neuralfolio
   ```

2. **Copy your files** into the cloned directory:
   ```bash
   # Copy the prepared files
   cp /path/to/your/app.py .
   cp /path/to/your/requirements.txt .
   cp /path/to/your/README.md .
   cp -r /path/to/your/checkpoints .
   cp /path/to/your/environment.py .
   cp /path/to/your/custom_policy.py .
   cp /path/to/your/llm_analysis_rag.py .
   cp /path/to/your/fetch_market_data.py .
   cp /path/to/your/.gitattributes .
   ```

3. **Track large files with LFS**:
   ```bash
   git lfs track "checkpoints/*.zip"
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Initial deploy: NeuralFolio Deep RL Portfolio Manager"
   git push
   ```

---

## Step 5: Monitor Build & Troubleshoot

1. Go to your Space URL: `https://huggingface.co/spaces/keshav775/neuralfolio`
2. Click the **"Building"** tab to watch the build logs
3. First build takes 5-10 minutes (installing PyTorch, transformers, etc.)

### Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **Out of memory** | Switch to `Qwen/Qwen2.5-1.5B-Instruct` or upgrade to GPU Space |
| **Build timeout** | Remove unnecessary dependencies from `requirements.txt` |
| **Module not found** | Ensure all `.py` files from `scripts/` are in root with fixed imports |
| **Checkpoint not loading** | Verify `.gitattributes` is correct and Git LFS tracked the file |

---

## Step 6: Upgrade Hardware (Optional)

If the free CPU tier is too slow:

1. Go to **Settings** in your Space
2. Under **Space Hardware**, upgrade to:
   - **T4 small** (GPU, $0.60/hour) — recommended for the 3B LLM
   - **A10G large** (faster GPU) — if you need real-time inference
3. Your Space will rebuild automatically

---

## Step 7: Share Your Space

Your live Space URL:  
**https://huggingface.co/spaces/keshav775/neuralfolio**

You can:
- Embed it in your portfolio website
- Share the link on LinkedIn/resume
- Enable the Discussions tab for feedback

---

## Local Testing Before Deploy

Test your restructured app locally:

```bash
cd /path/to/restructured/neuralfolio
python app.py
```

Visit `http://localhost:7860` — if it works locally, it will work on Spaces.

---

## Quick Checklist

- [ ] Space created on Hugging Face
- [ ] `app.py` in root with fixed imports
- [ ] All module files (`environment.py`, etc.) copied to root
- [ ] `requirements.txt` updated for free-tier (1.5B model)
- [ ] `.gitattributes` configured for Git LFS
- [ ] Checkpoint files tracked with `git lfs track`
- [ ] Files pushed to Space repository
- [ ] Build completed successfully
- [ ] App is live and functional

---

## Need Help?

- Hugging Face Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Deployment Guide: https://gradio.app/sharing-your-app/
- Community Forum: https://discuss.huggingface.co/

---

🎉 **Congratulations!** Your NeuralFolio app is now live on Hugging Face Spaces.
