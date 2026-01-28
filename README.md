# 🧠 **Arabic RAG Chatbot (Bosala AI)**

**Bosala AI** is an Arabic Retrieval-Augmented Generation (**RAG**) chatbot that answers Arabic questions by retrieving relevant content from a vector database and generating responses using **Google Gemini**. The project supports both a web interface and command-line tools, is fully **Dockerized**, and is designed to be easy to use even for users with no prior ML or backend experience. 

---

## **What This Project Does**
- Stores Arabic documents as semantic vectors (embeddings) in `Qdrant` collection.
- Searches for the most relevant content when a user asks a question
- Sends the retrieved context to `Gemini` to generate a high-quality Arabic answer
- Tracks usage and performance through an analytics dashboard
- Provides user accounts, chat history, and admin controls via a web app
- Allow administrators to direct the compass to any knowledge base, through the `ingest` interface that processes texts and stores them in the database

---

## **Key Features**
🇵🇸 Arabic-first RAG pipeline

🤖 Gemini 2.5 Flash for answer generation

📦 Qdrant vector database for semantic search

🌐 Django web application
  - User registration & login
  - Chat and Ingest UI
  - Admin & analytics dashboard

🐳 Fully Dockerized

🖥️ CLI tools for ingestion, chat, and evaluation

---
## **High-Level Architecture**
```
User Question → Embedding Model → Qdrant (Vector Search) → Relevant Context → Gemini (Answer Generation) → Final Arabic Answer
```
---
## **Main Components**

- ragchat/
  Core RAG logic: embeddings, retrieval, generation, evaluation, CLI tools

- backend/
  Django web app:
  - UI
  - User authentication
  - Analytics dashboard
  - API endpoints

- data/
  Stores raw data, processed data, and Qdrant vector storage
  
- load_testing/
  Uses locust script to test the RAG pipeline 

- Docker & Docker Compose
  Runs everything with a single command
---
## **Requirements**
- Docker
- Docker Compose
- Google Gemini API key

No Python, Django, or ML knowledge required to run ')

---
## **Setup (One Time)**
1. Clone the repository
    ```
    git clone https://github.com/your-org/rag_arabic_chatbot.git
    cd rag_arabic_chatbot
    ```
2. Create a .env file
    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    API_SECRET=random_secret
    EMB_MODEL=abdulrahman-nuzha/intfloat-multilingual-e5-large-arabic-fp16
    GEN_MODEL=models/gemini-2.5-flash
    GEN_MAX_NEW_TOKENS=512
    GEN_TEMPERATURE=0.4
    TOP_K=5
    ```
3. Running the Project

   a. Build everything
    ```
    docker compose build
    ```
   b. Start all services
    ```
    docker compose up
    ```
This will start:
  - Qdrant (vector database)
  - Django backend
  - RAG services

---
## **Technologies Used**
- Google Gemini
- Qdrant
- Django
- Docker
- Python
- Sentence Embeddings (Arabic)