import { streamChat } from "./api.js";
import { renderCitedHtml } from "./citations.js";

const STORAGE_KEY = "rag_chat_conversation_id";

const logEl = document.getElementById("chat-log");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const newBtn = document.getElementById("new-chat-btn");
const statusEl = document.getElementById("status");
const emptyEl = document.getElementById("empty-hint");
const refsListEl = document.getElementById("refs-list");
const refsEmptyEl = document.getElementById("refs-empty");

let conversationId = sessionStorage.getItem(STORAGE_KEY) || null;
let streaming = false;
/** @type {Map<string, {index:number, source:string, title:string, id:string}>} */
const references = new Map();

function setStatus(text, isError = false) {
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function hideEmpty() {
  if (emptyEl) emptyEl.hidden = true;
}

function refKey(ref) {
  return `${ref.index}:${ref.id}`;
}

function upsertReferences(refs) {
  let changed = false;
  for (const ref of refs) {
    const key = refKey(ref);
    if (!references.has(key)) {
      references.set(key, ref);
      changed = true;
    }
  }
  if (changed) renderReferencesPanel();
}

function renderReferencesPanel() {
  if (!refsListEl) return;
  refsListEl.innerHTML = "";
  const items = [...references.values()].sort((a, b) => a.index - b.index);
  if (refsEmptyEl) refsEmptyEl.hidden = items.length > 0;

  for (const ref of items) {
    const li = document.createElement("li");
    li.className = "ref-item";
    li.id = `ref-${ref.index}`;
    li.innerHTML = `
      <span class="ref-index">${ref.index}</span>
      <div class="ref-body">
        <div class="ref-title">${escapeText(ref.title)}</div>
        <div class="ref-meta"><span class="meta-pill">${escapeText(ref.source)}</span></div>
        <div class="ref-id" title="${escapeText(ref.id)}">${escapeText(ref.id)}</div>
      </div>
    `;
    refsListEl.appendChild(li);
  }
}

function escapeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function appendMessage(role, content, { error = false } = {}) {
  hideEmpty();
  const article = document.createElement("article");
  article.className = `msg ${error ? "error" : role}`;
  const label = document.createElement("span");
  label.className = "role";
  label.textContent = error ? "Error" : role === "user" ? "Tú" : "Asistente";
  const body = document.createElement("div");
  body.className = "body";
  if (role === "assistant" && !error) {
    const { html, refs } = renderCitedHtml(content);
    body.innerHTML = html;
    upsertReferences(refs);
  } else {
    body.textContent = content;
  }
  article.append(label, body);
  logEl.appendChild(article);
  logEl.scrollTop = logEl.scrollHeight;
  return body;
}

function updateAssistantBody(bodyEl, rawText) {
  const { html, refs } = renderCitedHtml(rawText);
  bodyEl.innerHTML = html;
  upsertReferences(refs);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearChat() {
  conversationId = null;
  sessionStorage.removeItem(STORAGE_KEY);
  logEl.querySelectorAll(".msg").forEach((el) => el.remove());
  references.clear();
  renderReferencesPanel();
  if (emptyEl) emptyEl.hidden = false;
  setStatus("");
  inputEl.focus();
}

newBtn.addEventListener("click", () => {
  if (streaming) return;
  clearChat();
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || streaming) return;

  streaming = true;
  sendBtn.disabled = true;
  newBtn.disabled = true;
  inputEl.value = "";
  setStatus("Generando…");

  appendMessage("user", message);
  const assistantBody = appendMessage("assistant", "");
  assistantBody.parentElement.classList.add("typing");
  let rawAssistant = "";

  try {
    await streamChat({
      message,
      conversationId,
      handlers: {
        onMeta(data) {
          if (data.conversation_id) {
            conversationId = data.conversation_id;
            sessionStorage.setItem(STORAGE_KEY, conversationId);
          }
          if (Array.isArray(data.sources)) {
            upsertReferences(data.sources);
          }
        },
        onToken(data) {
          rawAssistant += data.content || "";
          updateAssistantBody(assistantBody, rawAssistant);
        },
        onDone(data) {
          if (data.conversation_id) {
            conversationId = data.conversation_id;
            sessionStorage.setItem(STORAGE_KEY, conversationId);
          }
          updateAssistantBody(assistantBody, rawAssistant);
          setStatus("");
        },
        onError(data) {
          const text = data.error || "Error en el stream";
          if (!rawAssistant.trim()) {
            assistantBody.parentElement.className = "msg error";
            assistantBody.parentElement.querySelector(".role").textContent = "Error";
            assistantBody.textContent = text;
          } else {
            appendMessage("assistant", text, { error: true });
          }
          setStatus(text, true);
        },
      },
    });

    if (!rawAssistant.trim() && !assistantBody.textContent.trim()) {
      assistantBody.textContent = "(sin respuesta)";
    }
  } catch (err) {
    const text = err.message || "No se pudo completar el chat";
    if (!rawAssistant.trim()) {
      assistantBody.parentElement.className = "msg error";
      assistantBody.parentElement.querySelector(".role").textContent = "Error";
      assistantBody.textContent = text;
    } else {
      appendMessage("assistant", text, { error: true });
    }
    setStatus(text, true);
  } finally {
    assistantBody.parentElement.classList.remove("typing");
    streaming = false;
    sendBtn.disabled = false;
    newBtn.disabled = false;
    inputEl.focus();
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    formEl.requestSubmit();
  }
});

renderReferencesPanel();
