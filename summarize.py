from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Charge le modèle
model = AutoModelForSeq2SeqLM.from_pretrained("./t5-summarization-model")
tokenizer = AutoTokenizer.from_pretrained("./t5-summarization-model")
model.eval()

def generate_summary(text: str, max_length: int = 128) -> str:
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    article = """
    The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday,
    marking the tenth consecutive rate hike as the central bank continues its fight against
    inflation. Fed Chair Jerome Powell said the decision was unanimous among policymakers.
    The rate now stands at 5.25%, the highest level in 16 years. Markets reacted negatively,
    with the S&P 500 falling 1.2% following the announcement.
    """

    summary = generate_summary(article)
    print("=== Résumé ===")
    print(summary)
