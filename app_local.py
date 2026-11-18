import os
import re
import json
import html
import streamlit as st
import streamlit.components.v1 as components
import ollama

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "llama3.2:latest"

st.set_page_config(
    page_title="CYNBIT_VOICE_MENTOR",
    page_icon="🎙️",
    layout="wide"
)

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
# 3. SESSION STATE AND GLOBAL CONFIGURATION
# ==========================================

defaults = {
    "candidate_name": "Candidate",
    "role": "Machine Learning Engineer",
    "question": None,
    "feedback": None,
    "history": [],
    "interview_started": False,
    "audio_transcript": None,
    "last_audio_key": None
}

# Initialize session state safely
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Evaluation labels - available throughout the application
score_labels = {
    "technical_correctness": "Technical Correctness",
    "concept_understanding": "Concept Understanding",
    "clarity": "Communication Clarity",
    "completeness": "Answer Completeness"
}

# ==========================================
# 4. OLLAMA CONNECTION
# ==========================================

def get_ollama_response(prompt, system_prompt=None):

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        options={
            "temperature": 0.3,
            "num_predict": 700
        }
    )

    result = response["message"]["content"].strip()

    if not result:
        raise ValueError("Ollama returned an empty response.")

    return result


# ==========================================
# 5. GENERATE INTERVIEW QUESTION
# ==========================================

def generate_question(role, previous_questions):

    prompt = f"""
You are a professional technical interviewer.

Candidate role: {role}

Previously asked questions:
{json.dumps(previous_questions)}

Generate exactly ONE new technical interview question.

Requirements:
- Suitable for a junior or fresher candidate.
- Test practical technical understanding.
- Avoid repeating previous questions.
- Keep it clear and concise.
- Use a progressive difficulty level.
- Do not provide the answer.
- Do not include multiple questions.

Return only the question.
"""

    return get_ollama_response(
        prompt,
        "You are an experienced technical interviewer."
    )


# ==========================================
# 6. PARSE JSON RESPONSE
# ==========================================

def parse_json_response(response_text):

    cleaned_text = response_text.strip()

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

        match = re.search(
            r"\{.*\}",
            cleaned_text,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "Could not read structured AI feedback."
            )

        return json.loads(match.group())


# ==========================================
# 7. VALIDATE SCORES
# ==========================================

def validate_score(value):

    try:
        score = float(value)
    except (ValueError, TypeError):
        raise ValueError("Invalid evaluation score.")

    if not 0 <= score <= 10:
        raise ValueError("Score must be between 0 and 10.")

    return score


# ==========================================
# 8. LOCAL SPEECH-TO-TEXT
# ==========================================

@st.cache_resource
def load_whisper_model():

    from faster_whisper import WhisperModel

    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


def transcribe_audio(audio_file):

    model = load_whisper_model()

    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(audio_file.getvalue())
        temp_path = temp_file.name

    try:

        segments, info = model.transcribe(
            temp_path,
            beam_size=5
        )

        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        )

        if not transcript.strip():
            raise ValueError(
                "No speech detected. Please record again."
            )

        return transcript

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================
# 9. ANSWER EVALUATION
# ==========================================

def evaluate_answer(question, role, answer_content):

    prompt = f"""
You are a professional technical interview evaluator.

Candidate role: {role}

Interview question:
{question}

Candidate answer:
{answer_content}

Evaluate the answer using four dimensions:

1. Technical correctness
2. Concept understanding
3. Clarity
4. Completeness

Give each dimension a score from 0 to 10.

Evaluation rules:
- Be fair to junior and fresher candidates.
- Do not judge accent or speaking speed.
- Do not reward unnecessary long answers.
- Do not invent information the candidate did not provide.
- Evaluate technical accuracy carefully.
- Give constructive feedback.
- If the answer is incorrect, explain what is missing.
- If the answer is empty or unrelated, score accordingly.

Return ONLY valid JSON using this structure:

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
    "mentor_message": "An encouraging and honest message.",
    "next_step": "One practical suggestion."
}}

Important:
- Scores must be numeric values between 0 and 10.
- Strengths and improvements must be lists.
- Do not include Markdown outside the JSON.
"""

    raw_response = get_ollama_response(
        prompt,
        "You are a fair, accurate and supportive ML Engineer interview evaluator. Return valid JSON only."
    )

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
                f"Missing evaluation field: {field}"
            )

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

    overall_score = (
        scores["technical_correctness"] * 0.35
        + scores["concept_understanding"] * 0.25
        + scores["clarity"] * 0.20
        + scores["completeness"] * 0.20
    )

    evaluation["scores"] = scores
    evaluation["overall_score"] = round(
        overall_score, 1
    )

    return evaluation


# ==========================================
# 10. VOICE FEEDBACK
# ==========================================

def speak_feedback(feedback):

    speech_text = (
        "Your interview feedback. "
        + feedback["mentor_message"]
        + " Your next improvement step is. "
        + feedback["next_step"]
    )

    safe_text = json.dumps(speech_text)

    html_content = """
    <div style="display:flex;gap:10px;font-family:Arial;">

        <button id="speak"
        style="background:#167C70;color:white;
        border:1px solid #55D6BE;padding:11px 17px;
        border-radius:8px;cursor:pointer;">
            🔊 Speak Feedback
        </button>

        <button id="stop"
        style="background:#293B55;color:white;
        border:1px solid #40536D;padding:11px 17px;
        border-radius:8px;cursor:pointer;">
            Stop
        </button>

    </div>

    <script>

        const messageText = SPEECH_TEXT;

        document.getElementById("speak").onclick = function() {

            if (!window.speechSynthesis) {
                alert("Speech is not supported.");
                return;
            }

            window.speechSynthesis.cancel();

            const message =
                new SpeechSynthesisUtterance(messageText);

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

    html_content = html_content.replace(
        "SPEECH_TEXT",
        safe_text
    )

    components.html(
        html_content,
        height=55
    )


# ==========================================
# 11. SIDEBAR
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

    st.markdown("### Local AI Engine")

    st.success("Ollama: Llama 3.2")

    if st.button(
        "Reset Practice Session",
        use_container_width=True
    ):

        st.session_state.question = None
        st.session_state.feedback = None
        st.session_state.history = []
        st.session_state.interview_started = False
        st.session_state.audio_transcript = None
        st.session_state.last_audio_key = None

        st.rerun()


# ==========================================
# 12. HEADER
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
# 13. CANDIDATE DASHBOARD
# ==========================================

candidate_name = html.escape(
    st.session_state.candidate_name
)

st.markdown(
    f"## Welcome, {candidate_name}!"
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

col1.metric(
    "Questions Completed",
    total_questions
)

col2.metric(
    "Average Score",
    f"{average_score:.1f}/10"
    if average_score is not None else "—"
)

col3.metric(
    "Interview Role",
    st.session_state.role
)

col4.metric(
    "Practice Progress",
    f"{progress * 100:.0f}%"
)

st.progress(progress)

st.caption(
    f"{total_questions} of 10 practice questions completed"
)

st.divider()


# ==========================================
# 14. START INTERVIEW
# ==========================================

st.markdown("## Interview Workspace")

if not st.session_state.interview_started:

    st.markdown("""
    <div class="feedback-card">
        <h3 style="color:#F3F6FC;">Ready to practise?</h3>
        <p style="color:#A7B5C8;">
        Your AI interviewer will ask technical questions
        based on your selected role. Answer using your
        microphone or type your response.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Start Interview",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Preparing your first question using Llama 3.2..."
            ):

                question = generate_question(
                    st.session_state.role,
                    []
                )

            st.session_state.question = question
            st.session_state.feedback = None
            st.session_state.interview_started = True

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to start interview: {error}"
            )


# ==========================================
# 15. QUESTION AND ANSWER INPUT
# ==========================================

if st.session_state.interview_started:

    st.markdown("### Current Interview Question")

    safe_question = html.escape(
        st.session_state.question
    )

    st.markdown(
        f'<div class="question-card">'
        f'{safe_question}'
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
                    "Analyzing your answer with local AI..."
                ):

                    if audio_answer is not None:

                        transcript = transcribe_audio(
                            audio_answer
                        )

                        st.session_state.audio_transcript = transcript

                        if typed_answer.strip():
                            submitted_answer = (
                                typed_answer.strip()
                                + "\n\nAdditional voice response: "
                                + transcript
                            )
                        else:
                            submitted_answer = transcript

                    else:

                        submitted_answer = typed_answer.strip()

                    feedback = evaluate_answer(
                        st.session_state.question,
                        st.session_state.role,
                        submitted_answer
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
# 16. DISPLAY STRUCTURED FEEDBACK
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

    # Voice transcription
    if st.session_state.audio_transcript:

        st.markdown("### Voice Transcription")

        st.info(st.session_state.audio_transcript)

    # Evaluation breakdown
    st.markdown("### Evaluation Breakdown")

    for key, label in score_labels.items():

        value = feedback["scores"].get(key, 0)

        st.write(f"**{label} — {value}/10**")

        st.progress(value / 10)

    st.divider()

    # Strengths and improvements
    left_col, right_col = st.columns(2)

    with left_col:

        st.markdown("### What You Did Well")

        for point in feedback.get("strengths", []):
            st.success(point)

    with right_col:

        st.markdown("### Areas to Improve")

        for point in feedback.get("improvements", []):
            st.warning(point)

    # Improved answer
    st.markdown("### Improved Sample Answer")

    st.markdown(
        '<div class="feedback-card">'
        + html.escape(feedback.get("improved_answer", ""))
        + '</div>',
        unsafe_allow_html=True
    )

    # AI mentor
    st.markdown("### ✨ Your AI Mentor")

    st.markdown(
        '<div class="mentor-card">'
        '<p style="color:#55D6BE;font-weight:bold;">'
        'Mentor Message</p>'
        '<p style="color:#D3E2E8;">'
        + html.escape(feedback.get("mentor_message", ""))
        + '</p>'
        '<p style="color:#55D6BE;font-weight:bold;">'
        'Your Next Step</p>'
        '<p style="color:#D3E2E8;">'
        + html.escape(feedback.get("next_step", ""))
        + '</p></div>',
        unsafe_allow_html=True
    )

    # Voice feedback
    st.markdown("### 🔊 Listen to Your Feedback")

    speak_feedback(feedback)

    st.divider()

    # Next question
    if total_questions >= 10:

        st.success(
            "You have completed all 10 practice questions!"
        )

    else:

        if st.button(
            "Next Interview Question",
            use_container_width=True
        ):

            previous_questions = [
                item["question"]
                for item in st.session_state.history
            ]

            try:

                with st.spinner(
                    "Preparing your next question..."
                ):

                    next_question = generate_question(
                        st.session_state.role,
                        previous_questions
                    )

                st.session_state.question = next_question
                st.session_state.feedback = None
                st.session_state.audio_transcript = None

                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to generate next question: {error}"
                )

# ==========================================
# 17. INTERVIEW HISTORY
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

            st.markdown("**Evaluation Breakdown**")

            for key, label in score_labels.items():

                value = feedback["scores"].get(key, 0)

                st.write(f"{label}: {value}/10")

            st.markdown("**Mentor Message**")

            st.write(feedback.get("mentor_message", ""))
