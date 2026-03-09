from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI(title="Economic Summarization API")

model = AutoModelForSeq2SeqLM.from_pretrained("./t5-summarization-model")
tokenizer = AutoTokenizer.from_pretrained("./t5-summarization-model")
model.eval()

class TextInput(BaseModel):
    text: str
    max_length: int = 128

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Résumé Automatique</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #2c3e50; }
            textarea { width: 100%; height: 200px; padding: 10px; font-size: 14px; border-radius: 8px; border: 1px solid #ccc; }
            button { margin-top: 10px; padding: 10px 30px; background: #2c3e50; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 16px; }
            button:hover { background: #34495e; }
            #result { margin-top: 20px; padding: 15px; background: white; border-radius: 8px; border-left: 4px solid #2c3e50; display: none; }
            #loading { display: none; color: #888; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Résumé Automatique de Textes Economiques</h1>
        <p>Entrez votre texte ci-dessous et cliquez sur <b>Résumer</b> :</p>
        <textarea id="inputText" placeholder="Collez votre article ou rapport économique ici..."></textarea>
        <br>
        <button onclick="summarize()">Résumer</button>
        <p id="loading">Génération du résumé en cours...</p>
        <div id="result">
            <h3>Résumé :</h3>
            <p id="summaryText"></p>
            <small id="stats" style="color: #888;"></small>
        </div>

        <script>
            async function summarize() {
                const text = document.getElementById("inputText").value;
                if (!text.trim()) { alert("Veuillez entrer un texte !"); return; }

                document.getElementById("loading").style.display = "block";
                document.getElementById("result").style.display = "none";

                const response = await fetch("/summarize", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                document.getElementById("loading").style.display = "none";
                document.getElementById("result").style.display = "block";
                document.getElementById("summaryText").innerText = data.summary;
                document.getElementById("stats").innerText = `Mots originaux: ${data.original_length} | Mots résumé: ${data.summary_length}`;
            }
        </script>
    </body>
    </html>
    """

@app.post("/summarize")
def summarize(input: TextInput):
    inputs = tokenizer(
        "summarize: " + input.text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=input.max_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "original_length": len(input.text.split()),
        "summary_length": len(summary.split()),
        "summary": summary
    }