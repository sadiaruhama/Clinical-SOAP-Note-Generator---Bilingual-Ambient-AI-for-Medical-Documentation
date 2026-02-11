import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Configuration & Setup
model_id = "ruhameow/medical-scribe-weights"
subfolder = "final_medical_scribe_model"

# Load Tokenizer with fallback
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, use_fast=False)
except Exception as e:
    print(f"Loading local tokenizer failed, falling back to base: {e}")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=False)

# Load Model
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, subfolder=subfolder)

# 2. THE UPDATED FUNCTION GOES HERE
def generate_soap(dialogue):
    input_text = "summarize: " + dialogue
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=150,           # Strict limit to prevent rambling
            min_length=30,            # Ensure it's not too short
            num_beams=2,
            repetition_penalty=3.5,    # Stronger penalty for loops
            no_repeat_ngram_size=3,    # Prevents phrase repetition
            early_stopping=True
        )
    
    result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Simple cleanup of common artifacts
    clean_result = result.replace("", "").replace("“", '"').replace("”", '"')
    return clean_result

# 3. Gradio UI Setup (Points to the function above)
demo = gr.Interface(
    fn=generate_soap,
    inputs=gr.Textbox(lines=10, label="Input Clinical Dialogue", placeholder="Paste doctor-patient transcript here..."),
    outputs=gr.Textbox(label="Generated SOAP Note"),
    title="Clinical SOAP Note Generator",
    description="Ambient AI for structured medical summarization (SOAP format)."
)

if __name__ == "__main__":
    demo.launch()