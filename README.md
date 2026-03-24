# HR RAG System

## Overview
HR RAG System is a Retrieval-Augmented Generation (RAG) application for querying HR employee data using natural language. The system transforms structured employee data into searchable documents, retrieves relevant context using embeddings, and generates answers using an LLM.

## Architecture
HR Data → Document Processing → Embeddings (OpenAI) → Chroma Vector Store → Retrieval → LLM → FastAPI → Response

## Features
- Converts HR employee data into structured documents
- Generates embeddings using OpenAI
- Stores vectors in Chroma for efficient retrieval
- Uses LangChain for retrieval and response generation
- Exposes a FastAPI endpoint for querying the system
- Returns context-aware, generated answers from HR data

## Tech Stack
- Python  
- OpenAI  
- LangChain  
- Chroma (Vector Database)  
- FastAPI  

## Example Use Cases
- Summarize an employee profile  
- Query HR data using natural language  
- Retrieve insights from employee records  
- Generate answers grounded in internal data  

## API Usage

### Endpoint
```

POST /query

````

### Request Body
```json
{
  "question": "Give me a summary of employee 1"
}
````

### Response

```json
{
  "answer": "..."
}
```

## Deployment

This project is structured as an API-based AI system using FastAPI. It can be deployed to cloud platforms, but is currently provided as a local or development environment due to dataset size and deployment constraints.

## Project Structure

```
hr-rag-system/
├── hr_employee_rag.py        # RAG pipeline and core logic
├── hr_rag_api.py             # FastAPI application
├── hr_data/
│   └── employees/            # Processed employee documents
├── HR-Employee-Attrition.csv # Source dataset
├── requirements.txt          # Dependencies
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/moatazsaad/hr-rag-system.git
cd hr-rag-system
```

Create a virtual environment:

```bash
python -m venv env
```

Activate environment:

Windows:

```bash
env\Scripts\activate
```

macOS/Linux:

```bash
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Run the API:

```bash
uvicorn hr_rag_api:app --reload
```

## Example Query

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"question": "Give me a summary of employee 1"}'
```

## Project Highlights

* End-to-end RAG pipeline from structured HR data to generated answers
* Combines embeddings, vector database, and LLMs
* Exposes a production-style API using FastAPI
* Demonstrates applied GenAI and retrieval-based systems
