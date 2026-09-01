from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

import os
import base64
import requests
import json


# ==========================================================
# ENVIRONMENT
# ==========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MURF_API_KEY = os.getenv("MURF_API_KEY")


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "Interview API is running"
    })

# ==========================================================
# AI MODEL
# ==========================================================

model = init_chat_model(
    "google_genai:gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)


# ==========================================================
# INTERVIEW STATE
# ==========================================================

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer
)

question_count = 0
current_subject = ""

# For now we use a single session.
# Later this can be replaced with a user/session ID.
thread_id = "interview_session"


# ==========================================================
# INTERVIEW PROMPT
# ==========================================================

INTERVIEW_PROMPT = """You are Natalie, a friendly and conversational interviewer conducting a natural {subject} interview.

IMPORTANT GUIDELINES:

1. Ask exactly 5 questions total throughout the interview.
2. Keep questions SHORT and CRISP (1-2 sentences maximum).
3. ALWAYS reference what the candidate ACTUALLY said in their previous answer.
4. NEVER make up, assume, or invent anything about their answers.
5. Show genuine interest with brief acknowledgments based on their REAL responses.
6. Adapt questions based on their ACTUAL responses.
7. Go deeper when the candidate demonstrates strong knowledge.
8. Adjust difficulty when the candidate appears uncertain.
9. Be warm and conversational but CONCISE.
10. Do not give lengthy explanations.
11. Ask only one question at a time.

CRITICAL:
Read the conversation history carefully.
Only acknowledge information the candidate actually provided.

Keep the interview natural, short, conversational, and adaptive.
"""


# ==========================================================
# MURF TEXT-TO-SPEECH
# ==========================================================

def stream_audio(text):

    BASE_URL = "https://global.api.murf.ai/v1/speech/stream"

    payload = {
        "text": text,
        "voiceId": "en-US-natalie",
        "model": "FALCON",
        "multiNativeLocale": "en-US",
        "sampleRate": 24000,
        "format": "MP3",
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": MURF_API_KEY
    }

    response = requests.post(
        BASE_URL,
        headers=headers,
        data=json.dumps(payload),
        stream=True
    )

    response.raise_for_status()

    for chunk in response.iter_content(chunk_size=4096):

        if chunk:
            yield base64.b64encode(chunk).decode("utf-8") + "\n"


# ==========================================================
# START INTERVIEW
# ==========================================================

@app.route("/start-interview", methods=["POST"])
def start_interview():

    global question_count
    global current_subject
    global checkpointer
    global agent

    try:

        data = request.json or {}

        current_subject = data.get(
            "subject",
            "Python"
        )

        # First question
        question_count = 1

        # --------------------------------------------------
        # Create fresh memory for a new interview
        # --------------------------------------------------

        checkpointer = InMemorySaver()

        agent = create_agent(
            model=model,
            tools=[],
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        formatted_prompt = INTERVIEW_PROMPT.format(
            subject=current_subject
        )

        # --------------------------------------------------
        # Generate first question
        # --------------------------------------------------

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": formatted_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Start the interview with a warm greeting "
                            f"and ask the first question about "
                            f"{current_subject}. "
                            f"Keep it SHORT (1-2 sentences)."
                        )
                    }
                ]
            },
            config=config
        )

        question = response["messages"][-1].content

        print(
            f"\n[Question {question_count}] {question}"
        )

        # --------------------------------------------------
        # Stream audio
        # --------------------------------------------------

        return Response(
            stream_audio(question),
            content_type="text/plain"
        )

    except Exception as e:

        print(f"Start interview error: {e}")

        return jsonify({
            "success": False,
            "message": "Failed to start interview.",
            "error": str(e)
        }), 500


# ==========================================================
# SUBMIT ANSWER
# ==========================================================

@app.route("/submit-answer", methods=["POST"])
def submit_answer():

    global question_count
    global agent
    global checkpointer
    global current_subject

    try:

        # --------------------------------------------------
        # Check active interview
        # --------------------------------------------------

        if (
            agent is None
            or checkpointer is None
            or not current_subject
        ):

            return jsonify({
                "success": False,
                "message": "No active interview found."
            }), 400

        # --------------------------------------------------
        # Get candidate answer
        # --------------------------------------------------

        data = request.json or {}

        answer = data.get("answer", "").strip()

        if not answer:

            return jsonify({
                "success": False,
                "message": "Answer is required."
            }), 400

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # --------------------------------------------------
        # Check whether interview is already complete
        # --------------------------------------------------

        if question_count >= 5:

            return jsonify({
                "success": True,
                "interview_completed": True,
                "message": "Interview completed. Please request feedback."
            }), 200

        # --------------------------------------------------
        # Save candidate answer + generate next question
        # --------------------------------------------------

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": answer
                    }
                ]
            },
            config=config
        )

        # --------------------------------------------------
        # Increment question count
        # --------------------------------------------------

        question_count += 1

        next_question = response["messages"][-1].content

        print(
            f"\n[Candidate Answer] {answer}"
        )

        print(
            f"[Question {question_count}] {next_question}"
        )

        # --------------------------------------------------
        # Return next question as streamed audio
        # --------------------------------------------------

        return Response(
            stream_audio(next_question),
            content_type="text/plain"
        )

    except Exception as e:

        print(f"Submit answer error: {e}")

        return jsonify({
            "success": False,
            "message": "Failed to submit answer.",
            "error": str(e)
        }), 500


# ==========================================================
# GET FEEDBACK
# ==========================================================

@app.route("/feedback", methods=["GET"])
def get_feedback():

    global agent
    global checkpointer
    global current_subject
    global question_count

    # ------------------------------------------------------
    # Check active interview
    # ------------------------------------------------------

    if (
        agent is None
        or checkpointer is None
        or not current_subject
    ):

        return jsonify({
            "success": False,
            "message": "No active interview found."
        }), 400

    # ------------------------------------------------------
    # Check interview completion
    # ------------------------------------------------------

    if question_count < 5:

        return jsonify({
            "success": False,
            "message": "Interview is not completed yet.",
            "questions_completed": question_count,
            "questions_required": 5
        }), 400

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # --------------------------------------------------
        # Get conversation from LangGraph memory
        # --------------------------------------------------

        state = agent.get_state(config)

        messages = state.values.get(
            "messages",
            []
        )

        if not messages:

            return jsonify({
                "success": False,
                "message": "No interview conversation found."
            }), 400

        # --------------------------------------------------
        # Format conversation
        # --------------------------------------------------

        conversation_parts = []

        for message in messages:

            role = message.type.upper()

            content = message.content

            conversation_parts.append(
                f"{role}: {content}"
            )

        conversation = "\n".join(
            conversation_parts
        )

        # --------------------------------------------------
        # Feedback Prompt
        # --------------------------------------------------

        feedback_prompt = f"""
You are an expert technical interviewer.

The candidate has completed a 5-question
{current_subject} technical interview.

Analyze ONLY the candidate's actual answers
from the conversation below.

INTERVIEW CONVERSATION:

{conversation}

Generate a final interview evaluation.

Return ONLY valid JSON.
Do NOT use markdown.
Do NOT use code blocks.
Do NOT add any text outside the JSON.

Use exactly this structure:

{{
    "overall_score": 0,
    "summary": "",
    "strengths": [],
    "weaknesses": [],
    "technical_knowledge": "",
    "communication": "",
    "areas_to_improve": [],
    "recommendation": ""
}}

Evaluation rules:

1. overall_score:
   - Number between 0 and 10.
   - Judge the candidate's technical performance.

2. summary:
   - Give a concise overall assessment.
   - Keep it to 2-3 sentences.

3. strengths:
   - Give 2-4 specific strengths.
   - Use ONLY evidence from the candidate's answers.

4. weaknesses:
   - Give 2-4 specific weaknesses.
   - Do NOT invent weaknesses.

5. technical_knowledge:
   - Briefly assess the candidate's technical understanding.

6. communication:
   - Briefly assess clarity, confidence,
     structure and explanation quality.

7. areas_to_improve:
   - Give 2-4 concrete topics or skills
     the candidate should improve.

8. recommendation:
   - Give a concise interview/hiring recommendation.

IMPORTANT:

- Do not evaluate the interviewer.
- Do not evaluate the interviewer's questions.
- Do not make assumptions about candidate answers.
- Do not praise something the candidate did not demonstrate.
- Be honest but constructive.
"""

        # --------------------------------------------------
        # Generate feedback
        # --------------------------------------------------

        feedback_response = model.invoke(
            [
                {
                    "role": "system",
                    "content": feedback_prompt
                }
            ]
        )

        feedback_text = feedback_response.content.strip()

        # --------------------------------------------------
        # Remove markdown code fences if Gemini adds them
        # --------------------------------------------------

        if feedback_text.startswith("```"):

            feedback_text = feedback_text.replace(
                "```json",
                ""
            )

            feedback_text = feedback_text.replace(
                "```",
                ""
            )

            feedback_text = feedback_text.strip()

        # --------------------------------------------------
        # Parse JSON
        # --------------------------------------------------

        feedback = json.loads(
            feedback_text
        )

        # --------------------------------------------------
        # Return feedback
        # --------------------------------------------------

        return jsonify({
            "success": True,
            "subject": current_subject,
            "questions_completed": question_count,
            "feedback": feedback
        }), 200

    except json.JSONDecodeError:

        return jsonify({
            "success": False,
            "message": "AI returned an invalid feedback format."
        }), 500

    except Exception as e:

        print(
            f"Feedback error: {e}"
        )

        return jsonify({
            "success": False,
            "message": "Failed to generate interview feedback.",
            "error": str(e)
        }), 500


# ==========================================================
# RUN SERVER
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )