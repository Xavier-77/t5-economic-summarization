# Resume Automatique de Rapports Economiques avec T5

> Système de génération automatique de résumés pour l'analyse rapide de rapports économiques et financiers, basé sur un modèle T5 fine-tuné, avec une interface web interactive.

---

## Résultats

| Métrique | Score |
|----------|-------|
| ROUGE-1  | **0.3159** |
| ROUGE-2  | **0.1206** |
| ROUGE-L  | **0.2236** |

---

## Structure du projet

```
resume/
├── t5-summarization-model/     # Modèle T5 fine-tuné
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── main.py                     # API REST + Interface web (FastAPI)
├── summarize.py                # Script de résumé en ligne de commande
├── summarization.ipynb         # Notebook d'entraînement (Google Colab)
└── README.md
```

---

## Démarrage rapide

### 1. Cloner le dépôt
```bash
git clone https://github.com/Xavier-77/t5-economic-summarization.git
cd t5-economic-summarization
```

### 2. Installer les dépendances
```bash
pip install fastapi uvicorn transformers torch sentencepiece
```

### 3. Démarrer le serveur
```bash
uvicorn main:app --reload
```

Le serveur démarre sur : `http://127.0.0.1:8000`

Vous devriez voir dans le terminal :
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> Pour arrêter le serveur : CTRL + C

---

## Interface Web

Ouvrez votre navigateur et allez sur : http://127.0.0.1:8000

Vous verrez une interface où vous pouvez :
1. Coller ou taper votre texte économique
2. Cliquer sur **Résumer**
3. Obtenir le résumé automatique avec le nombre de mots

### Exemple

**Texte original (113 mots) :**
```
Apple Inc. reported record-breaking quarterly revenue of $97.3 billion in its fiscal
first quarter, surpassing analyst expectations of $94 billion. The tech giant saw strong
iPhone sales driven by the iPhone 15 lineup, which accounted for 58% of total revenue.
Services revenue also hit an all-time high of $23.1 billion, up 11% year-over-year...
```

**Résumé généré (35 mots) :**
```
Apple Inc. reported record-breaking quarterly revenue of $97.3 billion.
Services revenue also hit an all-time high of $23.1 billion, up 11% year-over-year.
Analysts remain bullish on the stock, with consensus price target of $210.
```

---

## Utilisation via API

### POST `/summarize`

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "The Federal Reserve raised interest rates by 0.25 percentage points..."}'
```

**Réponse :**
```json
{
  "original_length": 45,
  "summary_length": 18,
  "summary": "Federal Reserve raises interest rates by 0.25 percentage points."
}
```

### Interface interactive Swagger
Allez sur : http://127.0.0.1:8000/docs

---

## Modèle

- **Modèle de base** : `t5-small`
- **Tâche** : Génération de résumés (seq2seq)
- **Dataset** : [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) — 5 000 exemples d'entraînement
- **Entraînement** : 3 epochs, batch size 8, sur Google Colab (GPU T4)

---

## Métriques ROUGE

Les métriques ROUGE mesurent la qualité des résumés générés :

| Métrique | Description |
|----------|-------------|
| ROUGE-1 | Chevauchement des unigrammes entre résumé généré et référence |
| ROUGE-2 | Chevauchement des bigrammes |
| ROUGE-L | Plus longue sous-séquence commune |

---

## Stack technique

- **NLP** : HuggingFace Transformers, T5
- **Entraînement** : PyTorch, Seq2SeqTrainer
- **API + Interface** : FastAPI, Uvicorn
- **Données** : HuggingFace Datasets (CNN/DailyMail)
- **Évaluation** : rouge-score

---

## Pipeline d'entraînement

Le notebook complet est disponible dans `summarization.ipynb` :
1. Chargement du dataset CNN/DailyMail
2. Prétraitement et tokenisation avec le préfixe `summarize:`
3. Fine-tuning de T5 pour la génération de résumés
4. Évaluation avec les métriques ROUGE-1, ROUGE-2, ROUGE-L
5. Test sur des articles économiques et financiers
