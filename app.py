import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

# Language code mapping
LANGUAGE_CODES = {
    "English": "eng_Latn",
    "Myanmar": "mya_Mymr"
}

# Load models and tokenizers
model_1_path = "Henry922001/en-myan-finetuned-5000"
model_2_path = "Henry922001/en-myan-finetuned"

tokenizer_1 = AutoTokenizer.from_pretrained(model_1_path)
model_1 = AutoModelForSeq2SeqLM.from_pretrained(model_1_path)

tokenizer_2 = AutoTokenizer.from_pretrained(model_2_path)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_2_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_1 = model_1.to(device)
model_2 = model_2.to(device)

# Translation function with timing
def translate(text, src_lang, tgt_lang, model, tokenizer):
    src_code = LANGUAGE_CODES[src_lang]
    tgt_code = LANGUAGE_CODES[tgt_lang]
    tokenizer.src_lang = src_code

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    start = time.time()
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_code],
        max_length=256,
        num_beams=5,
        temperature=0.7
    )
    end = time.time()
    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    duration = round(end - start, 3)
    return output, duration

# BLEU score function
def compute_bleu(reference, hypothesis):
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)
    return round(score * 100, 2)  # percentage style

# Combined function
def parallel_translate(text, reference, src_lang, tgt_lang):
    output_1, time_1 = translate(text, src_lang, tgt_lang, model_1, tokenizer_1)
    output_2, time_2 = translate(text, src_lang, tgt_lang, model_2, tokenizer_2)

    bleu_1 = compute_bleu(reference, output_1) if reference else "N/A"
    bleu_2 = compute_bleu(reference, output_2) if reference else "N/A"

    return (
        output_1, f"{time_1}s", f"{bleu_1}",
        output_2, f"{time_2}s", f"{bleu_2}"
    )

# Gradio UI
with gr.Blocks(css="footer {visibility: hidden;}") as demo:
    gr.Markdown("# 🌍 Parallel Translator: English ⇄ Myanmar")

    with gr.Row():
        src = gr.Dropdown(["English", "Myanmar"], value="English", label="From")
        tgt = gr.Dropdown(["English", "Myanmar"], value="Myanmar", label="To")

    input_text = gr.Textbox(lines=3, placeholder="Enter sentence to translate...", label="Input")
    reference_text = gr.Textbox(lines=2, placeholder="(Optional) Paste reference translation...", label="Reference Translation")

    translate_btn = gr.Button("Click to Translate")

    gr.Markdown("## 🧠 Model 1")
    model_1_output = gr.Textbox(label="Model 1 Translation")
    model_1_time = gr.Textbox(label="Model 1 Inference Time", interactive=False)
    model_1_bleu = gr.Textbox(label="Model 1 BLEU Score", interactive=False)

    gr.Markdown("## 🧠 Model 2")
    model_2_output = gr.Textbox(label="Model 2 Translation")
    model_2_time = gr.Textbox(label="Model 2 Inference Time", interactive=False)
    model_2_bleu = gr.Textbox(label="Model 2 BLEU Score", interactive=False)

    translate_btn.click(
        fn=parallel_translate,
        inputs=[input_text, reference_text, src, tgt],
        outputs=[
            model_1_output, model_1_time, model_1_bleu,
            model_2_output, model_2_time, model_2_bleu
        ]
    )

demo.launch(share=True)
