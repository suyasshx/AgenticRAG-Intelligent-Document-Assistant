# AI-Powered RAG Intelligent Document Assistant

An AI-powered document question-answering system built using **FastAPI, LangChain, PostgreSQL, PGVector, HuggingFace Embeddings, and Groq LLMs**.

The system processes PDF documents, converts their content into searchable vector representations, retrieves relevant information using semantic search, and generates context-aware answers using a Large Language Model.

---

## 🚀 Overview

Traditional document search mainly relies on keyword matching, which can fail when the user's question uses different wording from the document.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that combines:

- PDF document ingestion
- Text extraction and cleaning
- Text chunking
- Vector embeddings
- PostgreSQL + PGVector
- Semantic similarity search
- Context-aware retrieval
- Large Language Model generation
- JWT-based authentication
- Conversational question answering

Instead of relying only on the LLM's pretrained knowledge, the system retrieves relevant information from the uploaded documents and provides it as context to the LLM before generating an answer.

---

## 🏗️ Architecture

```
                         User
                           |
                           v
                  +-----------------+
                  |     FastAPI     |
                  |    REST API     |
                  +--------+--------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
      +--------------+            +--------------+
      |     JWT      |            |  QA / Chat   |
      | Authentication|            |   Endpoint   |
      +--------------+            +------+-------+
                                        |
                                        v
                               +-----------------+
                               |   RAG Pipeline  |
                               +--------+--------+
                                        |
                         +--------------+--------------+
                         |                             |
                         v                             v
                  +-------------+              +-------------+
                  |  PGVector   |              |   Groq LLM  |
                  |  Retrieval  |              | Generation  |
                  +------+------+              +------+------+
                         |                             |
                         +--------------+--------------+
                                        |
                                        v
                                +---------------+
                                | Final Answer  |
                                +---------------+
```

---

## ✨ Features

### 1. PDF Document Ingestion

The system processes PDF documents through an automated ingestion pipeline.

The pipeline:
- Loads PDF documents
- Extracts text using Unstructured
- Cleans unnecessary whitespace
- Splits documents into smaller chunks
- Generates vector embeddings
- Stores vectors and metadata in PostgreSQL using PGVector

### 2. Semantic Search

The system uses vector embeddings instead of simple keyword matching.

Documents are converted into numerical vector representations and stored inside PostgreSQL using PGVector.

When a user asks a question, the system:
- Converts the question to an embedding
- Performs vector similarity search
- Retrieves relevant document chunks
- Provides them to the LLM as context

### 3. Retrieval-Augmented Generation

The core of the application is a Retrieval-Augmented Generation pipeline that:
- Processes user questions
- Retrieves relevant documents
- Combines context with the question
- Generates answers using Groq LLM

### 4. Conversational Question Answering

The RAG service supports conversation history, allowing follow-up questions to be interpreted using previous context.

### 5. JWT Authentication

The application implements OAuth2 password authentication with JWT access tokens for secure API access.

### 6. PostgreSQL + PGVector

- PostgreSQL is used as the application's database
- PGVector provides vector storage and similarity search capabilities
- Stores user data, application data, embeddings, and metadata

---

## 🧠 RAG Pipeline

### Stage 1 — Document Ingestion

```
PDF → Text Extraction → Text Cleaning → Text Chunking 
    → Embedding Generation → PGVector
```

### Stage 2 — Question Answering

```
User Question → FastAPI Endpoint → Authentication → Query Embedding 
             → PGVector Search → Relevant Chunks → Groq LLM → Generated Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend programming |
| FastAPI | REST API framework |
| LangChain | RAG orchestration |
| PostgreSQL | Database |
| PGVector | Vector similarity search |
| SQLModel | Database models |
| SQLAlchemy | Database connectivity |
| HuggingFace | Embedding generation |
| Groq | LLM inference |
| Unstructured | PDF/document processing |
| Tesseract | OCR support |
| Poppler | PDF processing utilities |
| Poetry | Dependency management |
| OAuth2 | Authentication |
| JWT | Access token authentication |

---

## 📁 Project Structure

```
AgenticRAG-Intelligent-Document-Assistant/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── login.py
│   │   │   │   └── qa.py
│   │   │   └── deps.py
│   │   ├── config/
│   │   │   └── ingestion.yml
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── crud/
│   │   │   └── user_crud.py
│   │   ├── ingestion/
│   │   │   ├── run.py
│   │   │   └── utils/
│   │   │       └── embedding_models.py
│   │   ├── models/
│   │   │   └── user_model.py
│   │   ├── schemas/
│   │   │   ├── chat_schema.py
│   │   │   └── ingestion_schema.py
│   │   ├── services/
│   │   │   └── rag_service.py
│   │   └── utils/
│   │       └── general_helpers.py
│   ├── init_db.py
│   ├── main.py
│   ├── pyproject.toml
│   └── poetry.lock
│
├── data/
│   ├── raw/
│   └── extraction/
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

Make sure the following are installed:
- Python 3.11
- Poetry
- PostgreSQL
- PGVector
- Poppler
- Tesseract OCR

The project currently requires: `Python >= 3.11, < 3.12`

### 1. Clone the Repository

```bash
git clone https://github.com/Suyasshx/AgenticRAG-Intelligent-Document-Assistant.git
cd AgenticRAG-Intelligent-Document-Assistant
```

### 2. Install Dependencies

Move into the backend directory:

```bash
cd backend
```

Install dependencies using Poetry:

```bash
poetry install
```

You can run commands through Poetry using:

```bash
poetry run <command>
```

---

## 🗄️ PostgreSQL Setup

Create a PostgreSQL database:

```bash
ragdb
```

Make sure PostgreSQL is running before starting the application.

The application uses PostgreSQL for persistent storage and PGVector for vector similarity search.

---

## 🔐 Environment Variables

Create a `.env` file inside the backend directory.

Example:

```
DB_NAME=ragdb
DB_USER=postgres
DB_PASS=your_database_password
DB_HOST=localhost
DB_PORT=5432

FIRST_SUPERUSER=admin@example.com
FIRST_SUPERUSER_PASSWORD=your_admin_password

SECRET_KEY_ACCESS_API=your_secret_key

GROQ_API_KEY=your_groq_api_key

LLM_MODEL=your_groq_model
LLM_TEMPERATURE=0.0

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Important:** Never commit your `.env` file or API keys to GitHub.

---

## 🗃️ Database Initialization

From the backend directory:

```bash
poetry run python app/init_db.py
```

This initializes the required database tables and configured user accounts.

---

## 📄 Add Documents

Place PDF documents inside:

```
data/raw/
```

Example:

```
data/
└── raw/
    └── research-paper.pdf
```

The extracted document data will be stored inside: `data/extraction/`

---

## 🔄 Run the Ingestion Pipeline

From the backend directory:

```bash
poetry run python app/ingestion/run.py
```

The pipeline will:
1. Find PDF files
2. Extract text
3. Clean the extracted content
4. Split the document into chunks
5. Generate embeddings
6. Store the embeddings in PGVector

Example output:
```
Loading PDF: research-paper.pdf
Extracted text saved to: data/extraction/research-paper.json
Loaded 1 documents
Created document chunks
Creating PGVector collection: docs
Documents successfully stored in PGVector
```

---

## 🚀 Run the Application

From the backend directory:

```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at:
- **API:** http://127.0.0.1:8000
- **Swagger Docs:** http://127.0.0.1:8000/docs

---

## 🔑 Authentication API

### Login

**POST** `/api/v1/login/access-token`

The endpoint uses OAuth2 password authentication.

Example form data:
```
username=admin@example.com
password=your_password
```

Successful response:
```json
{
    "access_token": "JWT_TOKEN",
    "token_type": "bearer"
}
```

Use the returned token for protected endpoints.

---

## 💬 Question Answering API

### Chat

**POST** `/api/v1/qa/chat`

The endpoint requires a valid JWT bearer token.

Request:
```json
{
    "message": "What is this document about?"
}
```

Example response:
```json
{
    "data": "The document discusses ..."
}
```

---

## 🧪 Example API Workflow

**Step 1 — Login**
```
POST /api/v1/login/access-token
```
Receive JWT Access Token

**Step 2 — Authorize**
```
Authorization: Bearer <access_token>
```

**Step 3 — Ask a Question**
```
POST /api/v1/qa/chat

{
    "message": "What is the main contribution of this paper?"
}
```

**Step 4 — Retrieval**
The system searches the PGVector docs collection for relevant document chunks.

**Step 5 — Generation**
The retrieved chunks are provided to the Groq LLM as context.

**Step 6 — Response**
The generated answer is returned through the FastAPI endpoint.

---

## 🔍 RAG Service

The core RAG implementation is located at:
```
backend/app/services/rag_service.py
```

The service combines:
- HuggingFace Embeddings
- PGVector
- Retriever
- History-Aware Retriever
- Document Chain
- Groq LLM

The retrieval chain also incorporates conversation history to improve retrieval for follow-up questions.

---

## 🔒 Security

The API uses:
- OAuth2 Password Flow
- JWT access tokens
- Password hashing
- Protected API dependencies
- Active-user validation
- Superuser validation

Protected endpoints require:
```
Authorization: Bearer <JWT>
```

Sensitive values such as:
- `GROQ_API_KEY`
- `DB_PASS`
- `SECRET_KEY_ACCESS_API`

are stored in environment variables.

---

## 📊 Example Use Case

A research paper is placed inside: `data/raw/`

The ingestion pipeline is executed:
```bash
poetry run python app/ingestion/run.py
```

The document is then:
```
Extracted → Cleaned → Chunked → Embedded → Stored in PGVector
```

A user can then ask:
> "What is the main contribution of this paper?"

The system retrieves the most relevant sections and sends them to the LLM.

Other possible questions include:
- What problem does this paper address?
- What dataset was used?
- What methodology was proposed?
- What are the main contributions?
- What were the experimental results?
- What limitations are discussed?

---

## 📌 Configuration

Document ingestion settings are configured in:
```
backend/app/config/ingestion.yml
```

Example:
```yaml
PATH_RAW_PDF: "data/raw"
PATH_EXTRACTION: "data/extraction"
COLLECTION_NAME: "docs"

PDF_PARSER: "Unstructured"

EMBEDDING_MODEL: "text-embedding-ada-002"

TOKENIZER_CHUNK_SIZE: 2000
TOKENIZER_CHUNK_OVERLAP: 200
```

---

## ⚠️ Troubleshooting

### Python Version

The project requires Python 3.11.

Check the Python version used by Poetry:
```bash
poetry run python --version
```

Expected: `Python 3.11.x`

### PostgreSQL Connection Error

Check that:
- PostgreSQL is running
- The `ragdb` database exists
- Database username and password are correct
- PostgreSQL is running on the configured port
- PGVector is available

### PDF Extraction Error

The ingestion pipeline requires PDF/OCR utilities.

Verify Poppler:
```bash
pdfinfo -v
```

Verify Tesseract:
```bash
tesseract --version
```

Both commands should return their respective version information.

### Authentication Error

If the API returns:
```json
{
    "detail": "Not authenticated"
}
```

Make sure the request contains:
```
Authorization: Bearer <access_token>
```

---

## 🔮 Future Improvements

Potential improvements include:
- Multi-document collections
- Document upload API
- Streaming LLM responses
- RAG evaluation metrics
- Hybrid keyword + vector retrieval
- Retrieval reranking
- Better conversation memory
- Citation-aware answers
- Document-level access control
- Background document processing
- Async ingestion
- Redis embedding caching
- Frontend chat interface
- Docker deployment
- Cloud deployment
- Automated testing
- CI/CD integration

---

## 🎯 Use Cases

The system can be adapted for:
- Research paper analysis
- Academic document assistants
- Technical documentation search
- Internal company knowledge bases
- Legal document search
- Product documentation
- Educational material
- PDF question answering
- Enterprise knowledge management

---

## 👨‍💻 Author

**Suyash Dhage**

GitHub: https://github.com/Suyasshx

---

## ⭐ Acknowledgements

This project uses several open-source technologies:
- FastAPI
- LangChain
- PostgreSQL
- PGVector
- HuggingFace
- Groq
- Unstructured
- SQLModel
- SQLAlchemy

---

## 📄 License

This project is intended for educational, research, and demonstration purposes.
