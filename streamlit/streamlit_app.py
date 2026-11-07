import os
import torch
import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    WatermarkingConfig,
    WatermarkDetector,
)

MODEL_NAME = "LiquidAI/LFM2.5-230M"

# Use physical cores; more threads than that usually slows CPU inference
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))


@st.cache_resource(show_spinner="Loading model...")
def load_everything():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    # Optional: int8 dynamic quantization of Linear layers, typically 1.5-2.5x faster on CPU
    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    watermark_config = WatermarkingConfig()
    detector = WatermarkDetector(
        model_config=model.config,
        device="cpu",
        watermarking_config=watermark_config,
    )
    return tokenizer, model, watermark_config, detector


tokenizer, model, watermark_config, detector = load_everything()


def generate_watermarked_text(input_text: str, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": "You are a text generator - you generate text based on user inputs."},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            watermarking_config=watermark_config,
        )

    new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def classify_watermarked(text: str):
    ids = tokenizer([text], return_tensors="pt").input_ids
    with torch.inference_mode():
        return detector(ids, return_dict=True)


# --- UI ---
if "classif_text" not in st.session_state:
    st.session_state.classif_text = ""

max_tokens = st.slider("Max new tokens", 16, 256, 128, step=16)

if st.button("generate"):
    with st.spinner("Generating..."):
        st.session_state.classif_text = generate_watermarked_text(f"generate a prompt that is {max_tokens} words long", max_tokens)

st.text_area("classification_text", key="classif_text")

if st.button("classify"):
    result = classify_watermarked(st.session_state.classif_text)
    st.markdown(f"Watermarked?: {result.prediction} - Confidence: {result.confidence}")