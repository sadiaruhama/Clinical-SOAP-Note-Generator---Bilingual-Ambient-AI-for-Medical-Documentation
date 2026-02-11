# AIMScribe Inference API Guide

This document explains how to programmatically interact with the AIMScribe model for clinical SOAP note generation.

## 1. API Endpoint
- **Base URL:** `https://ruhameow-medical-soap-scribe.hf.space/`
- **Hugging Face Path:** `ruhameow/medical-soap-scribe`

## 2. Technical Specifications
- **Framework:** Gradio Serverless Inference
- **Input Parameter:** `dialogue` (String) - Raw transcript of the medical conversation.
- **Output:** String formatted in SOAP (Subjective, Objective, Assessment, Plan) structure.

## 3. How to Use
To run the provided `test_api_client.py` script:

1. **Install the Client:**
   ```bash
   pip install gradio_client