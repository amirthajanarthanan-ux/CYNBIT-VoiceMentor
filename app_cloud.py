
import os
import re
import json
import time
import sqlite3
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(
    page_title="CYNBIT_VOICE_MENTOR",
    page_icon="🎙️",
    layout="wide"
)

if not API_KEY:
    st.error("Gemini API key is missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# ==========================================
# 2. MIDNIGHT AI THEME
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1020;
    color: #F3F6FC;
}

[data-testid="stSidebar"] {
    background-color: #111B2E;
    border-right: 1px solid #293B55;
}

[data-testid="stMetric"] {
    background-color: #172238;
    padding: 17px;
    border-radius: 12px;
    border: 1px solid #293B55;
}

[data-testid="stMetricLabel"] {
    color: #A7B5C8;
}

[data-testid="stMetricValue"] {
    color: #55D6BE;
}

.main-heading {
    font-size: 31px;
    font-weight: 750;
    color: #F3F6FC;
}

.sub-heading {
    color: #9BAEC5;
    font-size: 14px;
}

.question-card {
    background: #172940;
    border-left: 4px solid #55D6BE;
    padding: 22px;
    border-radius: 10px;
    font-size: 19px;
    line-height: 1.6;
}

.feedback-card {
    background: #142338;
    border: 1px solid #293B55;
    padding: 20px;
    border-radius: 12px;
}

.mentor-card {
    background: #142D32;
    border: 1px solid #28635C;
    padding: 18px;
    border-radius: 12px;
}

.stButton > button {
    background-color: #167C70;
    color: white;
    border: 1px solid #23998A;
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    background-color: #209889;
    color: white;
    border-color: #55D6BE;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #55D6BE;
}

.footer {
    color: #8193AA;
    text-align: center;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

defaults = {
    "candidate_name": "Candidate",
    "role": "Machine Learning Engineer",
    "question": None,
    "feedback": None,
    "history": [],
    "interview_started": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 4. GEMINI API FUNCTION
# ==========================================

def get_gemini_response(contents):

    for attempt in range(3):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )

            if response.text:
                return response.text

            raise ValueError("Gemini returned an empty response.")

        except Exception as error:

            message = str(error).upper()

            temporary_error = any(
                word in message
                for word in [
                    "503",
                    "UNAVAILABLE",
                    "429",
                    "RESOURCE_EXHAUSTED"
                ]
            )

            if temporary_error and attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                raise error

    raise RuntimeError("Gemini request failed.")


# ==========================================
# 5. GENERATE INTERVIEW QUESTION
# ==========================================

def generate_question(role, previous_questions):

    prompt = f"""
    You are a professional technical interviewer.

    Candidate role: {role}

    Previously asked questions:
    {previous_questions}

    Generate one new technical interview question.

    Requirements:
    - Suitable for a junior candidate.
    - Test practical technical understanding.
    - Avoid repeating previous questions.
    - Keep it clear and concise.

    Return only the interview question.
    """

    return get_gemini_response(prompt)


# ==========================================
# 6. PARSE GEMINI JSON RESPONSE
# ==========================================

def parse_json_response(response_text):

    cleaned_text = response_text.strip()

    # Remove Markdown code fences if Gemini includes them.
    cleaned_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned_text,
        flags=re.IGNORECASE
    )

    cleaned_text = re.sub(
        r"\s*```$",
        "",
        cleaned_text
    )

    try:
        return json.loads(cleaned_text)

    except json.JSONDecodeError:

        # Try to extract the JSON object from the response.
        match = re.search(
            r"\{.*\}",
            cleaned_text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "Could not read the structured AI feedback."
            )

        return json.loads(match.group())


# ==========================================
# 7. VALIDATE EVALUATION SCORES
# ==========================================

def validate_score(value):

    try:
        score = float(value)
    except (ValueError, TypeError):
        raise ValueError("AI returned an invalid evaluation score.")

    if not 0 <= score <= 10:
        raise ValueError("Score must be between 0 and 10.")

    return score


# ==========================================
# 8. STRUCTURED ANSWER EVALUATION
# ==========================================

def evaluate_answer(question, role, answer_content):

    prompt = f"""
    You are a professional technical interview evaluator.

    Candidate role: {role}

    Interview question:
    {question}

    Evaluate the candidate's answer.

    If the provided content is audio, first understand
    the spoken response.

    Evaluate these four dimensions:

    1. Technical correctness
       Is the technical information accurate?

    2. Concept understanding
       Does the candidate understand the underlying concept?

    3. Clarity
       Is the explanation understandable and organized?

    4. Completeness
       Does the answer cover the important points?

    Give each dimension a score between 0 and 10.

    Be fair to a fresher or junior candidate.
    Do not judge accent or speaking speed.
    Do not reward an answer just for being long.
    Do not invent information that the candidate did not say.

    Return ONLY valid JSON using this exact structure:

    {{
        "technical_correctness": 8,
        "concept_understanding": 7,
        "clarity": 8,
        "completeness": 6,
        "strengths": [
            "Correctly explained the main concept"
        ],
        "improvements": [
            "Include a practical example"
        ],
        "improved_answer": "A technically correct sample answer.",
        "mentor_message": "A genuine encouraging message.",
        "next_step": "One practical suggestion for improvement."
    }}

    Important:
    - All four scores must be numeric values from 0 to 10.
    - Strengths and improvements must be lists of short points.
    - Do not include Markdown outside the JSON.
    - If the answer is empty or unrelated, score it accordingly.
    """

    raw_response = get_gemini_response([
        prompt,
        answer_content
    ])

    evaluation = parse_json_response(raw_response)

    required_fields = [
        "technical_correctness",
        "concept_understanding",
        "clarity",
        "completeness",
        "strengths",
        "improvements",
        "improved_answer",
        "mentor_message",
        "next_step"
    ]

    for field in required_fields:
        if field not in evaluation:
            raise ValueError(
                f"Evaluation is missing the field: {field}"
            )

    # Validate every individual score.
    scores = {
        "technical_correctness": validate_score(
            evaluation["technical_correctness"]
        ),
        "concept_understanding": validate_score(
            evaluation["concept_understanding"]
        ),
        "clarity": validate_score(
            evaluation["clarity"]
        ),
        "completeness": validate_score(
            evaluation["completeness"]
        )
    }

    # Calculate overall score in Python.
    # Technical correctness receives the highest weight.
    overall_score = (
        scores["technical_correctness"] * 0.35
        + scores["concept_understanding"] * 0.25
        + scores["clarity"] * 0.20
        + scores["completeness"] * 0.20
    )

    evaluation["scores"] = scores
    evaluation["overall_score"] = round(overall_score, 1)

    return evaluation


# ==========================================
# 9. VOICE FEEDBACK
# ==========================================

def speak_feedback(feedback):

    speech_text = (
        "Your interview feedback. "
        + feedback["mentor_message"]
        + " Your next improvement step is. "
        + feedback["next_step"]
    )

    safe_text = json.dumps(speech_text)

    html = """
    <div style="display:flex; gap:10px; font-family:Arial;">
        <button id="speak"
        style="background:#167C70;color:white;border:1px solid #55D6BE;
        padding:11px 17px;border-radius:8px;cursor:pointer;">
            🔊 Speak Feedback
        </button>

        <button id="stop"
        style="background:#293B55;color:white;border:1px solid #40536D;
        padding:11px 17px;border-radius:8px;cursor:pointer;">
            Stop
        </button>
    </div>

    <script>
        const messageText = SPEECH_TEXT;

        document.getElementById("speak").onclick = function() {
            if (!window.speechSynthesis) {
                alert("Speech is not supported in this browser.");
                return;
            }

            window.speechSynthesis.cancel();

            const message = new SpeechSynthesisUtterance(messageText);
            message.lang = "en-IN";
            message.rate = 0.92;
            message.pitch = 1;

            window.speechSynthesis.speak(message);
        };

        document.getElementById("stop").onclick = function() {
            window.speechSynthesis.cancel();
        };
    </script>
    """

    html = html.replace("SPEECH_TEXT", safe_text)

    components.html(html, height=55)


# ==========================================
# 10. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🎙️ CYNBIT")
    st.caption("VOICE MENTOR")

    st.divider()

    st.markdown("### Candidate Profile")

    st.session_state.candidate_name = st.text_input(
        "Candidate Name",
        value=st.session_state.candidate_name
    )

    roles = [
        "Python Developer",
        "Data Analyst",
        "Data Scientist",
        "Machine Learning Engineer",
        "AI/ML Engineer"
    ]

    selected_role = st.selectbox(
        "Interview Role",
        roles,
        index=roles.index(st.session_state.role)
    )

    if selected_role != st.session_state.role:

        if st.session_state.interview_started:
            st.warning(
                "Reset the current session before changing roles."
            )
        else:
            st.session_state.role = selected_role

    st.divider()

    st.markdown("### Practice Guidelines")
    st.caption("01 - Speak naturally")
    st.caption("02 - Explain concepts clearly")
    st.caption("03 - Learn from mistakes")
    st.caption("04 - Improve with every attempt")

    st.divider()

    if st.button("Reset Practice Session", use_container_width=True):

        st.session_state.question = None
        st.session_state.feedback = None
        st.session_state.history = []
        st.session_state.interview_started = False

        st.rerun()


# ==========================================
# 11. HEADER
# ==========================================

st.markdown(
    '<p class="main-heading">CYNBIT_VOICE_MENTOR</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-heading">'
    'Your personal AI interview preparation workspace'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 12. CANDIDATE DASHBOARD
# ==========================================

st.markdown(
    f"## Welcome, {st.session_state.candidate_name}!"
)

st.caption(
    "Your next opportunity starts here. "
    "Build confidence one answer at a time."
)

history = st.session_state.history

total_questions = len(history)

scores = [
    item["feedback"]["overall_score"]
    for item in history
]

average_score = (
    sum(scores) / len(scores)
    if scores else None
)

progress = min(total_questions / 10, 1.0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Questions Completed", total_questions)

col2.metric(
    "Average Score",
    f"{average_score:.1f}/10"
    if average_score is not None else "—"
)

col3.metric("Interview Role", st.session_state.role)

col4.metric("Practice Progress", f"{progress * 100:.0f}%")

st.progress(progress)

st.caption(
    f"{total_questions} of 10 practice questions completed"
)

st.divider()


# ==========================================
# 13. START INTERVIEW
# ==========================================

st.markdown("## Interview Workspace")

if not st.session_state.interview_started:

    st.markdown("""
    <div class="feedback-card">
        <h3 style="color:#F3F6FC;">Ready to practise?</h3>
        <p style="color:#A7B5C8;">
        Your AI interviewer will ask technical questions based
        on your selected role. Answer using your microphone
        or type your response.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Start Interview",
        type="primary",
        use_container_width=True
    ):

        try:
            with st.spinner("Preparing your first question..."):

                question = generate_question(
                    st.session_state.role,
                    []
                )

            st.session_state.question = question
            st.session_state.feedback = None
            st.session_state.interview_started = True

            st.rerun()

        except Exception as error:
            st.error(f"Unable to start interview: {error}")


# ==========================================
# 14. QUESTION AND ANSWER INPUT
# ==========================================

if st.session_state.interview_started:

    st.markdown("### Current Interview Question")

    st.markdown(
        f'<div class="question-card">'
        f'{st.session_state.question}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🎤 Record Your Answer")

    audio_answer = st.audio_input(
        "Use your microphone",
        key=f"audio_{total_questions}"
    )

    if audio_answer:
        st.success("Recording captured successfully.")

    typed_answer = st.text_area(
        "Or type your answer",
        placeholder="Explain the concept in your own words...",
        height=120,
        key=f"answer_{total_questions}"
    )

    if st.button(
        "Evaluate My Answer",
        type="primary",
        use_container_width=True
    ):

        if audio_answer is None and not typed_answer.strip():

            st.warning(
                "Please record your answer or type a response."
            )

        else:

            try:

                with st.spinner(
                    "Analyzing your answer and preparing feedback..."
                ):

                    if audio_answer is not None:

                        answer_content = types.Part.from_bytes(
                            data=audio_answer.getvalue(),
                            mime_type=audio_answer.type or "audio/wav"
                        )

                        submitted_answer = "Voice recording submitted"

                    else:

                        answer_content = (
                            "Candidate's written answer: "
                            + typed_answer
                        )

                        submitted_answer = typed_answer

                    feedback = evaluate_answer(
                        st.session_state.question,
                        st.session_state.role,
                        answer_content
                    )

                st.session_state.feedback = feedback

                st.session_state.history.append({
                    "question": st.session_state.question,
                    "answer": submitted_answer,
                    "feedback": feedback
                })

                st.rerun()

            except Exception as error:

                st.error(
                    f"Evaluation failed: {error}"
                )


# ==========================================
# 15. DISPLAY STRUCTURED FEEDBACK
# ==========================================

if st.session_state.feedback:

    feedback = st.session_state.feedback

    st.divider()

    st.markdown("## Your Performance Review")

    score_col1, score_col2 = st.columns(2)

    score_col1.metric(
        "Overall Score",
        f"{feedback['overall_score']}/10"
    )

    score_col2.metric(
        "Questions Completed",
        total_questions
    )

    st.markdown("### Evaluation Breakdown")

    score_labels = {
        "technical_correctness": "Technical Correctness",
        "concept_understanding": "Concept Understanding",
        "clarity": "Clarity",
        "completeness": "Completeness"
    }

    for key, label in score_labels.items():

        value = feedback["scores"][key]

        st.write(f"**{label} — {value}/10**")

        st.progress(value / 10)

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:

        st.markdown("### What You Did Well")

        for point in feedback["strengths"]:
            st.success(point)

    with right_col:

        st.markdown("### Areas to Improve")

        for point in feedback["improvements"]:
            st.warning(point)

    st.markdown("### Improved Sample Answer")

    st.markdown(
        f'<div class="feedback-card">'
        f'{feedback["improved_answer"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("### ✨ Your AI Mentor")

    st.markdown(
        f'<div class="mentor-card">'
        f'<p style="color:#55D6BE;font-weight:bold;">'
        f'Mentor Message'
        f'</p>'
        f'<p style="color:#D3E2E8;">'
        f'{feedback["mentor_message"]}'
        f'</p>'
        f'<p style="color:#55D6BE;font-weight:bold;">'
        f'Your Next Step'
        f'</p>'
        f'<p style="color:#D3E2E8;">'
        f'{feedback["next_step"]}'
        f'</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("### 🔊 Listen to Your Feedback")

    speak_feedback(feedback)

    st.divider()

    if st.button(
        "Next Interview Question",
        use_container_width=True
    ):

        previous_questions = [
            item["question"]
            for item in st.session_state.history
        ]

        try:

            with st.spinner("Preparing your next question..."):

                next_question = generate_question(
                    st.session_state.role,
                    previous_questions
                )

            st.session_state.question = next_question
            st.session_state.feedback = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to generate the next question: {error}"
            )


# ==========================================
# 16. INTERVIEW HISTORY
# ==========================================

if st.session_state.history:

    st.divider()

    st.markdown("## Your Interview History")

    for index, item in enumerate(
        st.session_state.history,
        start=1
    ):

        feedback = item["feedback"]

        with st.expander(
            f"Question {index} | "
            f"Score: {feedback['overall_score']}/10"
        ):

            st.markdown("**Question**")
            st.write(item["question"])

            st.markdown("**Candidate Answer**")
            st.write(item["answer"])

            st.markdown("**Evaluation**")

            for key, label in score_labels.items():

                st.write(
                    f"{label}: {feedback['scores'][key]}/10"
                )

            st.markdown("**Mentor Message**")
            st.write(feedback["mentor_message"])


# ==========================================
# 17. FOOTER
# ==========================================

st.divider()

st.markdown("""
<p class="footer">
CYNBIT_VOICE_MENTOR
<br>
Practice. Learn. Improve.
<br>
Built to help candidates grow with confidence.
</p>
""", unsafe_allow_html=True)
