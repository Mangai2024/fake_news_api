import torch
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer from Hugging Face Model Hub
MODEL_NAME = "Mangai2024/fake-news-distilbert"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Fake News ❌" if prediction == 0 else "Real News ✅"

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Enter news text here..."),
    outputs="text",
    title="Fake News Detection",
    description="Enter a news article to check whether it is Fake or Real."
)

interface.launch()
