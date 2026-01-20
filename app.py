import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

st.set_page_config(page_title="Fake News Detection")

st.title("📰 Fake News Detection")
st.write("Enter a news article to check whether it is Fake or Real.")

@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("fake_news_model")
    tokenizer = DistilBertTokenizer.from_pretrained("fake_news_model")
    return model, tokenizer

model, tokenizer = load_model()

text = st.text_area("Enter news text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
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

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
