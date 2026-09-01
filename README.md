# 🎙️ AI Interview Assistant — Backend

Flask backend for an AI-powered technical interview assistant. It uses **Google Gemini** to conduct adaptive interviews, **LangGraph** to maintain conversation state, and **Murf AI** for voice responses.

## Features

* AI-driven technical interviews
* Adaptive questions based on candidate responses
* 5-question interview flow
* Voice responses using Murf AI
* Conversation memory with LangGraph
* AI-generated performance feedback and scoring

## Tech Stack

* Python
* Flask
* Google Gemini 2.5 Flash
* LangChain / LangGraph
* Murf AI
* Flask-CORS

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key
MURF_API_KEY=your_murf_api_key
```

### 3. Run the server

```bash
python app.py
```

## API

| Method | Endpoint           | Description                                       |
| ------ | ------------------ | ------------------------------------------------- |
| `POST` | `/start-interview` | Starts a new interview                            |
| `POST` | `/submit-answer`   | Submits an answer and generates the next question |
| `GET`  | `/feedback`        | Generates final interview feedback                |

### Start Interview

```json
POST /start-interview

{
  "subject": "Python"
}
```

### Submit Answer

```json
POST /submit-answer

{
  "answer": "Python lists are mutable collections."
}
```

### Get Feedback

```text
GET /feedback
```

Returns structured feedback including:

* Overall score
* Summary
* Strengths
* Weaknesses
* Technical knowledge
* Communication
* Areas to improve
* Recommendation

> **Note:** The current implementation uses in-memory conversation storage and is intended for development/demo use.

Live demo here :- https://my-interview-assistant.vercel.app/
