import {
  createStudySession,
  finishStudySession,
  getStudyProfile,
  listDocuments,
  regenerateStudySummary,
  streamChat,
  submitStudyAnswer,
} from "./api.js";
import { renderCitedHtml } from "./citations.js";
import { createConversationId } from "./conversations.js";

const notesListEl = document.getElementById("notes-list");
const notesEmptyEl = document.getElementById("notes-empty");
const placeholderEl = document.getElementById("study-placeholder");
const overviewEl = document.getElementById("study-overview");
const studyTitleEl = document.getElementById("study-title");
const studyStatusEl = document.getElementById("study-status");
const studySummaryEl = document.getElementById("study-summary");
const studyConceptsEl = document.getElementById("study-concepts");
const studyChatEl = document.getElementById("study-chat");
const studyChatLogEl = document.getElementById("study-chat-log");
const studyChatFormEl = document.getElementById("study-chat-form");
const studyInputEl = document.getElementById("study-message-input");
const studyQuizEl = document.getElementById("study-quiz");
const quizTitleEl = document.getElementById("quiz-title");
const quizProgressEl = document.getElementById("quiz-progress");
const quizQuestionEl = document.getElementById("quiz-question");
const quizAnswerEl = document.getElementById("quiz-answer");
const quizFeedbackEl = document.getElementById("quiz-feedback");
const quizReportEl = document.getElementById("quiz-report");
const btnSubmitAnswer = document.getElementById("btn-submit-answer");
const btnFinishExam = document.getElementById("btn-finish-exam");
const statusEl = document.getElementById("status");

/** @type {string | null} */
let selectedDocId = null;
/** @type {{ id: string, filename: string } | null} */
let selectedDoc = null;
/** @type {string | null} */
let studyConversationId = null;
/** @type {import("./api.js").StudySession | null} */
let activeSession = null;
/** @type {number} */
let currentQuestionIndex = 0;
/** @type {"quiz" | "exam" | null} */
let activeQuizMode = null;
let streaming = false;

function setStatus(text, isError = false) {
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function hideAllViews() {
  placeholderEl.hidden = true;
  overviewEl.hidden = true;
  studyChatEl.hidden = true;
  studyQuizEl.hidden = true;
}

function showPlaceholder() {
  hideAllViews();
  placeholderEl.hidden = false;
}

function showOverview() {
  hideAllViews();
  overviewEl.hidden = false;
}

function showChat() {
  hideAllViews();
  studyChatEl.hidden = false;
}

function showQuiz() {
  hideAllViews();
  studyQuizEl.hidden = false;
}

async function loadNotes() {
  setStatus("Cargando apuntes…");
  try {
    const data = await listDocuments("notes");
    const docs = data.documents || [];
    notesListEl.innerHTML = "";
    if (notesEmptyEl) notesEmptyEl.hidden = docs.length > 0;

    for (const doc of docs) {
      const li = document.createElement("li");
      li.className = "conversation-item";
      if (doc.id === selectedDocId) li.classList.add("active");
      li.innerHTML = `
        <button type="button" class="conversation-link" data-id="${escapeHtml(doc.id)}">
          <span class="conversation-title">${escapeHtml(doc.filename)}</span>
          <span class="conversation-meta">${escapeHtml(doc.format)} · ${doc.chunk_count} chunks</span>
        </button>
      `;
      li.querySelector("button")?.addEventListener("click", () => selectDocument(doc));
      notesListEl.appendChild(li);
    }
    setStatus("");
  } catch (err) {
    setStatus(err.message || "Error cargando apuntes", true);
  }
}

async function selectDocument(doc) {
  selectedDocId = doc.id;
  selectedDoc = doc;
  studyConversationId = createConversationId();
  await loadNotes();
  await loadProfile();
}

async function loadProfile() {
  if (!selectedDocId || !selectedDoc) {
    showPlaceholder();
    return;
  }
  showOverview();
  studyTitleEl.textContent = selectedDoc.filename;
  studySummaryEl.textContent = "Cargando resumen…";
  studyConceptsEl.innerHTML = "";
  studyStatusEl.textContent = "…";

  try {
    const profile = await getStudyProfile(selectedDocId);
    studyStatusEl.textContent = profile.status;
    if (profile.status === "ready") {
      studySummaryEl.innerHTML = renderCitedHtml(profile.summary || "(Sin resumen)");
      studyConceptsEl.innerHTML = (profile.key_concepts || [])
        .map((c) => `<li>${escapeHtml(c)}</li>`)
        .join("");
    } else if (profile.status === "pending") {
      studySummaryEl.textContent = "Generando resumen… Recarga en unos segundos.";
    } else {
      studySummaryEl.textContent = profile.error || "Error al generar el resumen.";
    }
    setStatus("");
  } catch (err) {
    if (err.status === 404) {
      studySummaryEl.textContent =
        "El resumen aún no está disponible. Puede estar procesándose tras la ingesta.";
      studyStatusEl.textContent = "pending";
    } else {
      setStatus(err.message || "Error cargando perfil", true);
    }
  }
}

function appendChatMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg msg--${role}`;
  div.innerHTML =
    role === "assistant"
      ? renderCitedHtml(content)
      : `<p>${escapeHtml(content)}</p>`;
  studyChatLogEl.appendChild(div);
  studyChatLogEl.scrollTop = studyChatLogEl.scrollHeight;
}

studyChatFormEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedDocId || streaming) return;
  const message = studyInputEl.value.trim();
  if (!message) return;

  appendChatMessage("user", message);
  studyInputEl.value = "";
  streaming = true;
  setStatus("Pensando…");

  const assistantEl = document.createElement("div");
  assistantEl.className = "msg msg--assistant";
  studyChatLogEl.appendChild(assistantEl);

  let fullText = "";
  try {
    await streamChat({
      message,
      conversationId: studyConversationId,
      mode: "study",
      documentId: selectedDocId,
      handlers: {
        onToken: (data) => {
          fullText += data.content || "";
          assistantEl.innerHTML = renderCitedHtml(fullText);
          studyChatLogEl.scrollTop = studyChatLogEl.scrollHeight;
        },
        onDone: () => setStatus(""),
        onError: (data) => setStatus(data.error || "Error en el chat", true),
      },
    });
  } catch (err) {
    setStatus(err.message || "Error en el chat", true);
  } finally {
    streaming = false;
  }
});

async function startSession(mode) {
  if (!selectedDocId) return;
  setStatus(`Preparando ${mode === "quiz" ? "test" : "examen"}…`);
  try {
    activeSession = await createStudySession({
      documentId: selectedDocId,
      mode,
    });
    activeQuizMode = mode;
    currentQuestionIndex = 0;
    quizReportEl.hidden = true;
    quizFeedbackEl.hidden = true;
    btnFinishExam.hidden = mode !== "exam";
    quizTitleEl.textContent = mode === "quiz" ? "Test" : "Examen";
    showQuiz();
    renderCurrentQuestion();
    setStatus("");
  } catch (err) {
    setStatus(err.message || "No se pudo crear la sesión", true);
  }
}

function renderCurrentQuestion() {
  if (!activeSession?.questions?.length) return;
  const question = activeSession.questions[currentQuestionIndex];
  quizProgressEl.textContent = `${currentQuestionIndex + 1} / ${activeSession.questions.length}`;
  quizQuestionEl.textContent = question.question;
  quizAnswerEl.value = question.user_answer || "";
  quizFeedbackEl.hidden = true;
  quizFeedbackEl.innerHTML = "";

  const evaluated = question.score != null;
  if (activeQuizMode === "quiz" && evaluated) {
    showQuestionFeedback(question);
  }
  btnSubmitAnswer.disabled = evaluated && activeQuizMode === "quiz";
}

function showQuestionFeedback(question) {
  quizFeedbackEl.hidden = false;
  quizFeedbackEl.innerHTML = `
    <p><strong>Puntuación:</strong> ${question.score}/10</p>
    <p>${escapeHtml(question.feedback || "")}</p>
  `;
}

btnSubmitAnswer.addEventListener("click", async () => {
  if (!activeSession) return;
  const question = activeSession.questions[currentQuestionIndex];
  const answer = quizAnswerEl.value.trim();
  if (!answer) {
    setStatus("Escribe una respuesta", true);
    return;
  }
  setStatus("Evaluando…");
  try {
    const result = await submitStudyAnswer(activeSession.id, question.id, answer);
    question.user_answer = answer;
    question.score = result.score;
    question.feedback = result.feedback;

    if (activeQuizMode === "quiz") {
      showQuestionFeedback({ ...question, feedback: result.feedback, score: result.score });
      btnSubmitAnswer.disabled = true;
      setStatus(`Puntuación: ${result.score}/10`);
      setTimeout(() => goNextQuestion(), 1500);
    } else {
      setStatus("Respuesta guardada");
      goNextQuestion();
    }
  } catch (err) {
    setStatus(err.message || "Error evaluando", true);
  }
});

function goNextQuestion() {
  if (!activeSession) return;
  if (currentQuestionIndex < activeSession.questions.length - 1) {
    currentQuestionIndex += 1;
    renderCurrentQuestion();
    btnSubmitAnswer.disabled = false;
    setStatus("");
    return;
  }
  if (activeQuizMode === "exam") {
    setStatus("Todas las preguntas respondidas. Pulsa Finalizar examen.");
  } else {
    finishQuizSession();
  }
}

async function finishQuizSession() {
  if (!activeSession) return;
  setStatus("Calculando nota…");
  try {
    const result = await finishStudySession(activeSession.id);
    activeSession = result.session;
    quizReportEl.hidden = false;
    quizReportEl.innerHTML = `
      <h4>Informe final — Nota: ${result.session.score}/10</h4>
      <pre>${escapeHtml(result.report)}</pre>
    `;
    quizQuestionEl.textContent = "Sesión completada";
    quizAnswerEl.hidden = true;
    btnSubmitAnswer.hidden = true;
    btnFinishExam.hidden = true;
    setStatus("");
  } catch (err) {
    setStatus(err.message || "Error finalizando", true);
  }
}

btnFinishExam.addEventListener("click", () => finishQuizSession());

document.getElementById("btn-repaso").addEventListener("click", () => {
  studyChatLogEl.innerHTML = "";
  studyConversationId = createConversationId();
  showChat();
});

document.getElementById("btn-test").addEventListener("click", () => startSession("quiz"));
document.getElementById("btn-exam").addEventListener("click", () => startSession("exam"));

document.getElementById("btn-back-overview").addEventListener("click", showOverview);
document.getElementById("btn-back-quiz").addEventListener("click", () => {
  activeSession = null;
  quizAnswerEl.hidden = false;
  btnSubmitAnswer.hidden = false;
  showOverview();
});

document.getElementById("btn-regen").addEventListener("click", async () => {
  if (!selectedDocId) return;
  setStatus("Regenerando resumen…");
  try {
    await regenerateStudySummary(selectedDocId);
    await loadProfile();
    setStatus("Resumen regenerado");
  } catch (err) {
    setStatus(err.message || "Error regenerando", true);
  }
});

loadNotes();
