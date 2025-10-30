# 🧠 Arabic RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot for Arabic question answering.  
It retrieves relevant Arabic passages from the **ARCD dataset** and generates answers using **Arabic GPT-2 (`aubmindlab/aragpt2-base`)**.  
The system is modular, well-structured, and fully **Dockerized**, allowing real-time interaction through the command line.

---

## 🚀 Features
- **Arabic text preprocessing & normalization**
- **Semantic embeddings** using multilingual Arabic embedding model  (`abdulrahman-nuzha/intfloat-multilingual-e5-large-arabic-fp16`)
- **Vector storage & retrieval** using **Qdrant**
- **Arabic GPT-2 answer generation** (`aubmindlab/aragpt2-base`)
- **Evaluation metrics:** `BLEU` and `F1`
- **Docker-based reproducibility**
- **CLI tools** for each pipeline step

---

## 🏗️ Project Structure
```plaintext
rag_arabic_chatbot/
│
├── ragchat/
│ ├── init.py
│ ├── cli.py # Load and preprocess ARCD dataset
│ ├── preprocessing.py # Arabic normalization and cleaning
│ ├── tokenization.py # Tokenization logic
│ ├── embeddings.py # Sentence embeddings
│ ├── qdrant_index.py # Qdrant database interface
│ ├── retriever.py # Retrieval logic from Qdrant
│ ├── generator.py # GPT-2 based answer generator
│ ├── pipeline.py # Combined RAG pipeline (retrieval + generation)
│ ├── embed_contexts_cli.py # CLI for embedding contexts
│ ├── embed_answers_cli.py # CLI for embedding answers
│ ├── search_cli.py # CLI to test retrieval
│ ├── chat_cli.py # Interactive chatbot
│ ├── evaluate_cli.py # BLEU/F1 evaluation
│ └── config.py # Paths, models, constants
│
├── data/
│ ├── raw/ # Raw dataset
│ ├── processed/ # Cleaned/tokenized dataset
│ └── qdrant_storage/ # Persistent Qdrant storage
│
├── Dockerfile # Image definition for ragapp
├── docker-compose.yml
├── requirements.txt # Dependencies
└── README.md
```
---
## ⚙️ How to Run
### 1️⃣ Build the project
```bash
docker compose build
```
### 2️⃣ Start Qdrant
```bash
docker compose up -d qdrant
```
### 3️⃣ Embed and store data
```bash
docker compose run --rm ragapp python -m ragchat.embed_contexts_cli
docker compose run --rm ragapp python -m ragchat.embed_answers_cli
```
### 4️⃣ Chat with the model
```bash
docker compose run --rm chat
```

### 5️⃣ Evaluate model (BLEU & F1)
```
docker compose run --rm evaluate
```
Example output:
``` 
Evaluating 50 samples from data/processed/arcd_clean_prepared ...
BLEU: 0.03
F1:   0.007
```
---
## 📊 Current Results
---

|   Metric  |   Value   |
|-----------|-----------|
|   BLEU    |   0.03    |
|    F1     |   0.007   |

- Note: Low values are expected because aubmindlab/aragpt2-base is a general Arabic LM, not instruction-tuned for factual QA.
- The focus of this project is building a functional RAG pipeline architecture for Arabic language tasks.

---
## 🧩 System Architecture
**1. Dataset Loading**
Loads the `ARCD` (Arabic Reading Comprehension Dataset) from Hugging Face (~1400 QA pairs).

**2. Preprocessing & Tokenization**
Applies Arabic text normalization, removes unwanted symbols, and tokenizes inputs.

**3. Embeddings Generation**
Encodes passages and answers using
`abdulrahman-nuzha/intfloat-multilingual-e5-large-arabic-fp16.`

**4. Vector Database (Qdrant)**
Stores embeddings and enables semantic search for top-k relevant passages.

**5. Retrieval**
For each question, retrieves the top relevant contexts based on cosine similarity.

**6. Prompt Construction**
Combines the retrieved contexts and the question into a concise Arabic prompt.

**7. Answer Generation**
The prompt is passed to `aubmindlab/aragpt2-base` for generating an Arabic response.

**8. Evaluation**
`BLEU` and `F1` scores are computed between generated and reference answers.