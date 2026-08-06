# scripts/llm_analysis_rag.py
import gc
import time
import os
import shutil
import torch
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta

# LangChain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.docstore.document import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
# HF_EMBEDDING_MODEL is no longer used in this reduced scope
HF_GENERATION_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# Global variables
llm_pipeline_hf_instance = None

# --- Helper: Robust JSON Extractor ---
def extract_clean_json(response_text):
    """Robust JSON extractor handling Python booleans and Markdown."""
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        text_to_parse = json_match.group(1)
    else:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text_to_parse = response_text[start_idx:end_idx+1]
        else:
            # print(f"❌ PARSE ERROR: No JSON found: {response_text[:100]}...")
            return None

    text_to_parse = text_to_parse.replace(": True", ": true").replace(": False", ": false")

    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError:
        return None

# --- Shared LLM Setup (Singleton Pattern) ---
def setup_llm_pipeline():
    global llm_pipeline_hf_instance
    if llm_pipeline_hf_instance is None:
        print(f"--- Loading Model: {HF_GENERATION_MODEL} ---")
        tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL, trust_remote_code=True)
        # 4-bit quantization config for efficient loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_GENERATION_MODEL, trust_remote_code=True,
            quantization_config=bnb_config, device_map="auto"
        )
        # Create the HF pipeline
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=1024, # Increased token limit for detailed historical analysis
            do_sample=False, temperature=0.1, # Low temp for factual responses
            return_full_text=False
        )
        llm_pipeline_hf_instance = HuggingFacePipeline(pipeline=pipe)
    return llm_pipeline_hf_instance

# =========================================
# NEW FUNCTION: Structured Historical Analysis
# =========================================
def analyze_historical_segment(df_segment, selected_assets, period_name):
    """
    Analyzes a specific segment of historical data directly without RAG.
    Takes a DataFrame slice, calculates summary stats, and prompts the LLM.
    """
    llm = setup_llm_pipeline()
    print(f"--- Running Historical Analysis for {period_name} ---")

    # 1. Create quantitative summary of the data segment for the prompt
    if df_segment.empty:
         return "No data available for this period to analyze."

    start_date = df_segment.index.min().date()
    end_date = df_segment.index.max().date()
    
    start_vals = df_segment.iloc[0]
    end_vals = df_segment.iloc[-1]
    # Calculate percentage change over the period, handling potential zeros
    pct_changes = ((end_vals - start_vals) / (start_vals.replace(0, np.nan)) * 100).fillna(0)
    
    # Build the context string
    data_summary = f"Analysis Period: {period_name} ({start_date} to {end_date})\n\n"
    data_summary += "Performance Summary over Period:\n"
    for asset in selected_assets:
        if asset in df_segment.columns:
            change = pct_changes[asset]
            direction = "gained" if change > 0 else "lost"
            data_summary += f"- {asset}: {direction} {abs(change):.2f}%\n"

    # Add volatility context (standard deviation of daily returns)
    data_summary += "\nVolatility Context (Daily Return Std Dev):\n"
    daily_rets = df_segment.pct_change()
    std_devs = daily_rets.std() * 100
    for asset in selected_assets:
         if asset in std_devs.index:
             data_summary += f"- {asset}: {std_devs[asset]:.2f}%\n"

    # 2. Create the Prompt
    # We use Qwen's chat template format (<|im_start|>...)
    prompt_template = """<|im_start|>system
You are a senior financial analyst. Your job is to analyze historical market data trends for selected assets over a specific time period.
Provide a concise, professional, and insightful summary of the performance, key trends, and comparative movements based *only* on the provided data summary.
Highlight significant gains, losses, or differences in volatility between the assets.

### DATA CONTEXT:
{data_summary}
<|im_end|>
<|im_start|>user
Generate the historical analysis report.
<|im_end|>
<|im_start|>assistant
"""
    pt = PromptTemplate(template=prompt_template, input_variables=["data_summary"])
    formatted_prompt = pt.format(data_summary=data_summary)

    # 3. Invoke LLM
    response = llm.invoke(formatted_prompt)
    return response.strip()


# --- Decision Analysis (Kept for Forecast Tab) ---
def analyze_agent_decision(current_market_data_window, proposed_allocations):
    """
    HYBRID ANALYZER: Python does the math, LLM does the talking.
    """
    llm = setup_llm_pipeline()

    # --- 1. PREPARE DATA ---
    # (Logic remains the same as before...)
    vix_level = current_market_data_window['VIX'].iloc[-1] if 'VIX' in current_market_data_window else 0

    # Identify largest position
    risky_assets = {k:v for k,v in proposed_allocations.items() if k not in ['Cash', 'TLT']}
    if risky_assets:
        max_asset = max(risky_assets, key=risky_assets.get)
        max_val = risky_assets[max_asset] * 100
    else:
        max_asset = "None"
        max_val = 0.0

    safe_haven_pct = (proposed_allocations.get('Cash', 0) + proposed_allocations.get('TLT', 0)) * 100

    # --- 2. PYTHON LOGIC CORE ---
    trigger_safe_haven = safe_haven_pct > 80.0
    trigger_crash_rule = vix_level > 20.0 and safe_haven_pct < 30.0
    trigger_concentration = vix_level > 15.0 and max_val > 40.0

    # Determine Verdict Programmatically
    calculated_risk = "MODERATE"
    reason_code = "Standard market conditions."

    if trigger_safe_haven:
        calculated_risk = "LOW"
        reason_code = f"Safe Haven Exception triggered (Safe Assets: {safe_haven_pct:.1f}% > 80%)."
    elif trigger_crash_rule:
        calculated_risk = "HIGH"
        reason_code = f"Crash Protocol triggered (VIX {vix_level:.1f} > 20 and Safe Haven < 30%)."
    elif trigger_concentration:
        calculated_risk = "HIGH"
        reason_code = f"Concentration Rule triggered (VIX {vix_level:.1f} > 15 and {max_asset} > 40%)."

    # --- 3. THE "NARRATOR" PROMPT ---
    prompt_template = """<|im_start|>system
You are a Senior Risk Analyst.
The Quantitative Engine has already processed the data and determined the Risk Level.
Your job is to summarize the strategy and explain the risk verdict to the user.

### QUANTITATIVE ENGINE OUTPUT:
- **Determined Risk Level:** {calculated_risk}
- **Primary Logic Trigger:** {reason_code}

### DATA CONTEXT:
- VIX: {vix:.2f}
- Largest Position: {max_asset} ({max_val:.1f}%)
- Safe Haven Allocation: {safe_pct:.1f}%

### INSTRUCTIONS:
1. **Strategy Summary:** Describe the allocation style (e.g., "Aggressive Tech", "Defensive Cash").
2. **Justification:** Explain the Risk Level using the "Primary Logic Trigger" provided above. Do not invent new math.

Return ONLY raw JSON:
{{
    "strategy_summary": "string",
    "risk_level": "{calculated_risk}",
    "justification": "string",
    "confidence_score": 10
}}
<|im_end|>
<|im_start|>user
Generate the report.
<|im_end|>
<|im_start|>assistant
"""
    pt = PromptTemplate(template=prompt_template, input_variables=["calculated_risk", "reason_code", "vix", "max_asset", "max_val", "safe_pct"])

    formatted = pt.format(
        calculated_risk=calculated_risk,
        reason_code=reason_code,
        vix=vix_level,
        max_asset=max_asset,
        max_val=max_val,
        safe_pct=safe_haven_pct
    )

    res = llm.invoke(formatted)
    return extract_clean_json(res)

# --- MAIN (for testing) ---
if __name__ == '__main__':
    print("Running test...")
    # Generate Dummy Data
    dates = pd.date_range(start="2023-01-01", periods=180, freq='D')
    df_dummy = pd.DataFrame({
        'SPY': np.linspace(400, 450, 180) + np.random.normal(0, 5, 180),
        'BTC-USD': np.linspace(30000, 40000, 180) + np.random.normal(0, 1000, 180),
        'VIX': np.linspace(20, 15, 180)
    }, index=dates)

    # Test the new historical analysis function
    selected = ['SPY', 'BTC-USD']
    period = "6 Months"
    print(f"\nTesting analysis for {selected} over {period}...")
    analysis = analyze_historical_segment(df_dummy, selected, period)
    print("\n--- LLM Analysis Output ---")
    print(analysis)