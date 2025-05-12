# 🏥 Named Entity Recognition (NER) in Healthcare Data

This project focuses on building a **custom Named Entity Recognition (NER)** model using **Conditional Random Fields (CRF)** to extract **diseases and treatments** from biomedical texts. It uses classical NLP techniques and a CRF-based sequence tagging pipeline, achieving **90.67% weighted F1-score** on the test data.

---

## 🚀 Objective

To accurately identify medical **entities** (Diseases - `D`, Treatments - `T`) from unstructured clinical documents and associate **diseases with their respective treatments** using a CRF-based model trained on POS-tag and context-aware features.

---

## 👨‍⚕️ Who Should Use This

- **Data Scientists**: Looking to apply CRFs and contextual features in healthcare NLP.
- **ML/AI Engineers**: Working on entity recognition, medical chatbot pipelines, or RAG systems.
- **Healthcare Analysts**: Extract structured insights from messy medical text.
- **GenAI Developers**: Preprocessing corpora for LLM training or fine-tuning.

---

## 🧠 Problem Statement

Unstructured medical records contain critical information such as diagnoses, medications, procedures, and more. The aim is to extract structured data (Diseases and Treatments) from such text using:
- POS-tag enriched features
- CRF model for sequence tagging
- Rule-based mapping for disease-treatment linkage

---

## 📦 Tech Stack

| Component       | Tool Used                         |
|----------------|-----------------------------------|
| Programming     | Python 3.9                        |
| NLP Model       | spaCy (`en_core_web_sm`)          |
| ML Framework    | `sklearn-crfsuite` (CRF)          |
| POS Tagging     | spaCy-based contextual POS        |
| Feature Design  | Custom handcrafted features       |
| Evaluation      | F1-score (weighted)               |
| Output Format   | CSV, Python Dict, Pandas DataFrame|

---

## 📊 Features Extracted for CRF

- `word.lower` → Lowercased word
- `word[-3:]`, `word[-2:]` → Last characters
- `word.isupper`, `word.isdigit`
- `word.startsWithCapital`
- `word.pos_tag` (contextual PoS tag)
- Previous word features (if available)
- Start/End of sentence flags

---

## 📁 Dataset Structure

Input format is **word-per-line**, with sentences separated by blank lines. Labels are provided in the same format.

**Sample:**
All O
live O
births O


After preprocessing, the dataset is structured as:
- 2,599 training sentences
- 1,056 testing sentences
- Over 48,000 training tokens

---

## 🛠️ Model Pipeline

1. **Data Preprocessing**
   - Load `.txt` files
   - Convert to full sentences
   - Contextual PoS tagging with spaCy

2. **Feature Engineering**
   - Apply handcrafted features per word
   - Add contextual PoS and neighbor word info

3. **Model Training**
   - Trained `sklearn-crfsuite.CRF` with 300 iterations
   - Achieved **90.67% F1-score** on the test set

4. **Entity Association**
   - Extract `(Disease → Treatment)` relationships from CRF outputs

---

## 🧪 Evaluation

- **Model**: Conditional Random Field (CRF)
- **Metric**: F1 Score (weighted)
- **Score**: `90.67%`

---

## 🧬 Output Example

```json
"Disease": "hereditary retinoblastoma",
"Treatment": ["radiotherapy"]
```
## 🔮 Future Improvements
Migrate to BiLSTM-CRF or transformer-based models (e.g., BioBERT)

Expand to include more entity types (e.g., Symptoms, Tests)

Visualize predictions with NER highlighting UI

Deploy as a REST API for real-time extraction

## 📂 File Organization
```bash
├── train_sent, train_label
├── test_sent, test_label
├── notebook.ipynb / script.py
├── Cleaned_Dict.pkl / .csv
├── requirements.txt
└── README.md
