
# 🩺 Clinical SOAP Note Generator
> **Bilingual Ambient AI for Medical Summarization**

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Model](https://img.shields.io/badge/Model-BART--Large--CNN-orange)
![Framework](https://img.shields.io/badge/Framework-HuggingFace-yellow)
![Status](https://img.shields.io/badge/Status-Deployment_Active-green)


## 👤 Personal Information
**Name:** Sadia Ruhama  
**Contact:** 01631029458  
**Email:** [sadia.ruhama@g.bracu.ac.bd](mailto:sadia.ruhama@g.bracu.ac.bd)  





## 📖 1.Project Overview
AIMScribe is an automated clinical documentation system that transforms raw, bilingual
(Bangla-English) doctor-patient transcripts into structured **SOAP (Subjective, Objective, Assessment,
Plan)** notes. By capturing clinical data in real-time, the tool aims to reduce the "documentation burden"
that leads to physician burnout and medical errors.

## 🧠2. Thought Process & Problem-Solving Approach
### Phase 1: Problem Definition
The task is a specialized form of **Abstractive Summarization**. Unlike general news summarization,
medical notes require Entity Extraction (medications, vitals) and **Structural Mapping** (putting the right
fact in the right SOAP category).
### Phase 2: Architectural Selection

I selected the **BART-Large-CNN** (Bidirectional and Auto-Regressive Transformers) architecture.

    ● **Why BART?** BART uses a bidirectional encoder (like BERT) to grasp the full context of a
patient's story and an autoregressive decoder (like GPT) to generate structured text. This makes it
significantly more effective than "decoder-only" models for capturing the specific nuances of
clinical dialogue.
### Phase 3: Complexity & Challenges
During the development, several critical complexities arose:
* ** 1. Linguistic Diversity: Patients in South Asia often switch between Bangla and English. To handle
this, I utilized the SentencePiece tokenizer to ensure that the Bangla script did not result in
"unknown token" errors.
* ** 2. Repetitive Hallucination: In early testing, the model tended to loop medical phrases. I mitigated
this by implementing a high Repetition Penalty (3.5) and no-repeat n-gram size of 3 during
inference.

* ** 3. VRAM Constraints: Fine-tuning a 400M+ parameter model on a T4 GPU required memory
engineering. I implemented Gradient Accumulation (to simulate a larger batch size) and FP16
Mixed Precision to prevent out-of-memory crashes.
3. Fine-Tuning Process
The model was fine-tuned on the Medical
Chat
_
_
Summarization dataset using the Hugging Face Trainer
API.
●
●
●
●
Epochs: 3
Learning Rate: 3e-5
Optimizer: AdamW
Strategy: The training utilized Gradient Checkpointing, which saves memory by recomputing
certain activations during the backward pass rather than storing them all.
4. Evaluation Results
Quantitative Performance (Epoch 2 Results)
The model was evaluated using ROUGE scores to measure the overlap between generated notes and
ground-truth references.
Metric Score Justification
ROUGE-1 0.5672 High capture of individual clinical keywords (vitals, symptoms).
ROUGE-L 0.4517 Primary Metric. Measures the longest common subsequence, proving the
model follows the correct SOAP structural flow.
Qualitative Analysis
●
Baseline Performance: Before training, the model produced general paragraphs without any
S-O-A-P headers, often missing specific medication dosages.
●
Fine-Tuned Success: After training, the model successfully isolated "Salmonella enterica
infection" into the Assessment section and "Cefazolin" into the Plan.
●
Failure Analysis: In cases with extremely long transcripts, the model occasionally truncated the
"Plan.
" To fix this, I expanded the max
_
length during the generation phase to 256 tokens.
5. Setup & API Usage Guide
Installation
Bash
pip install -r requirements.txt
API Usage (via Python)
The model is served via a Gradio API. You can query it programmatically as follows:
Python
from gradio
_
client import Client
client = Client("ruhameow/medical-soap-scribe")
result = client.predict(
dialogue="Patient reports lower back pain with stiffness.
"
,
api
_
name="/predict"
)
print(result)
6. Libraries Used
●
●
●
●
●
Transformers (Hugging Face): For BART model loading and Seq2Seq training.
Datasets & Evaluate: For processing clinical data and computing ROUGE scores.
Gradio: To build the front-end deployment interface.
PyTorch: The underlying deep learning engine for tensor computations.
SentencePiece: Critical for handling the Bangla-English bilingual tokenization.
7. Deployment Link: https://ruhameow-medical-soap-scribe.hf.space/
