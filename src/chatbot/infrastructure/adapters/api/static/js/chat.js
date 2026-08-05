import { getConversation, listModels, streamChat } from "./api.js";
import { renderCitedHtml } from "./citations.js";
import {
  createConversationId,
  getActiveConversationId,
  loadConversationIndex,
  removeConversation,
  setActiveConversationId,
  setTitleFromFirstQuery,
  upsertConversationMeta,
} from "./conversations.js";
import { initVoiceInput } from "./voice.js";

const logEl = document.getElementById("chat-log");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const voiceBtn = document.getElementById("voice-btn");
const voiceAutoSendEl = document.getElementById("voice-auto-send");
const newBtn = document.getElementById("new-chat-btn");
const statusEl = document.getElementById("status");
const emptyEl = document.getElementById("empty-hint");
const refsListEl = document.getElementById("refs-list");
const refsEmptyEl = document.getElementById("refs-empty");
const conversationsListEl = document.getElementById("conversations-list");
const conversationsEmptyEl = document.getElementById("conversations-empty");
const flowEls = document.querySelectorAll('input[name="retrieval-backend"]');
const modelSelectEl = document.getElementById("model-select");

let conversationId = getActiveConversationId();
let streaming = false;
/** @type {{ refresh?: () => void } | null} */
let voiceControls = null;
/** @type {Map<string, {index:number, source:string, title:string, id:string}>} */
const references = new Map();

function setStatus(text, isError = false) {
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function hideEmpty() {
  if (emptyEl) emptyEl.hidden = true;
}

function showEmpty() {
  if (emptyEl) emptyEl.hidden = false;
}

function escapeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function ensureActiveConversation() {
  if (conversationId) return conversationId;
  conversationId = createConversationId();
  setActiveConversationId(conversationId);
  upsertConversationMeta({ id: conversationId });
  renderConversationsPanel();
  return conversationId;
}

function startNewConversation() {
  conversationId = createConversationId();
  setActiveConversationId(conversationId);
  upsertConversationMeta({ id: conversationId });
  clearChatView();
  renderConversationsPanel();
  inputEl.focus();
}

function clearChatView() {
  logEl.querySelectorAll(".msg").forEach((el) => el.remove());
  references.clear();
  renderReferencesPanel();
  showEmpty();
  setStatus("");
}

function upsertReferences(refs) {
  let changed = false;
  for (const ref of refs) {
    if (!references.has(ref.id)) {
      references.set(ref.id, ref);
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

function renderConversationsPanel() {
  if (!conversationsListEl) return;
  const items = loadConversationIndex();
  conversationsListEl.innerHTML = "";
  if (conversationsEmptyEl) conversationsEmptyEl.hidden = items.length > 0;

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "conversation-item";
    if (item.id === conversationId) li.classList.add("active");

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "conversation-btn";
    btn.title = item.title;
    btn.textContent = item.title;
    btn.addEventListener("click", () => {
      if (streaming || item.id === conversationId) return;
      void switchConversation(item.id);
    });

    const del = document.createElement("button");
    del.type = "button";
    del.className = "conversation-delete";
    del.title = "Eliminar de la lista";
    del.setAttribute("aria-label", "Eliminar conversación");
    del.textContent = "×";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      if (streaming) return;
      removeConversation(item.id);
      if (conversationId === item.id) {
        const next = loadConversationIndex()[0];
        if (next) void switchConversation(next.id);
        else startNewConversation();
      } else {
        renderConversationsPanel();
      }
    });

    li.append(btn, del);
    conversationsListEl.appendChild(li);
  }
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
  return { article, body };
}

function ensureThinkingBlock(article) {
  let details = article.querySelector(".thinking-block");
  if (details) return details;
  details = document.createElement("details");
  details.className = "thinking-block";
  details.open = true;
  const summary = document.createElement("summary");
  summary.className = "thinking-summary";
  summary.innerHTML =
    '<span class="thinking-label">Pensando…</span><span class="thinking-hint">ver razonamiento</span>';
  const pre = document.createElement("pre");
  pre.className = "thinking-body";
  details.append(summary, pre);
  const body = article.querySelector(".body");
  article.insertBefore(details, body);
  return details;
}

function appendThinking(article, chunk) {
  const details = ensureThinkingBlock(article);
  const pre = details.querySelector(".thinking-body");
  pre.textContent += chunk;
  logEl.scrollTop = logEl.scrollHeight;
}

function finishThinking(article) {
  const details = article.querySelector(".thinking-block");
  if (!details) return;
  details.open = false;
  const label = details.querySelector(".thinking-label");
  if (label) label.textContent = "Pensamiento";
}

function updateAssistantBody(bodyEl, rawText) {
  const { html, refs } = renderCitedHtml(rawText);
  bodyEl.innerHTML = html;
  upsertReferences(refs);
  logEl.scrollTop = logEl.scrollHeight;
}

async function switchConversation(id) {
  conversationId = id;
  setActiveConversationId(id);
  upsertConversationMeta({ id });
  clearChatView();
  renderConversationsPanel();
  setStatus("Cargando conversación…");

  try {
    const data = await getConversation(id);
    const messages = Array.isArray(data.messages) ? data.messages : [];
    if (!messages.length) {
      showEmpty();
      setStatus("");
      return;
    }
    for (const msg of messages) {
      appendMessage(msg.role === "user" ? "user" : "assistant", msg.content);
    }
    const firstUser = messages.find((msg) => msg.role === "user");
    if (firstUser?.content) {
      setTitleFromFirstQuery(id, firstUser.content);
      renderConversationsPanel();
    }
    setStatus("");
  } catch (err) {
    // Conversación nueva aún no persistida en el servidor.
    if (err.status === 404) {
      showEmpty();
      setStatus("");
      return;
    }
    setStatus(err.message || "No se pudo cargar la conversación", true);
  }
}

function selectedBackend() {
  const checked = [...flowEls].find((el) => el.checked);
  return checked?.value || "postgres";
}

function selectedModel() {
  return modelSelectEl?.value?.trim() || null;
}

async function loadModelSelector() {
  if (!modelSelectEl) return;
  modelSelectEl.disabled = true;
  try {
    const data = await listModels();
    const models = Array.isArray(data.models) ? data.models : [];
    const active = data.active || models[0] || "";
    modelSelectEl.innerHTML = "";
    const options = models.length ? models : active ? [active] : [];
    for (const id of options) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = id;
      if (id === active) option.selected = true;
      modelSelectEl.appendChild(option);
    }
    if (!options.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Sin modelos";
      modelSelectEl.appendChild(option);
    }
  } catch (err) {
    modelSelectEl.innerHTML = "";
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No se pudieron cargar modelos";
    modelSelectEl.appendChild(option);
    setStatus(err.message || "No se pudieron cargar los modelos", true);
  } finally {
    if (!streaming) modelSelectEl.disabled = false;
  }
}

newBtn.addEventListener("click", () => {
  if (streaming) return;
  startNewConversation();
});

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message || streaming) return;

  const activeId = ensureActiveConversation();
  const meta = loadConversationIndex().find((item) => item.id === activeId);
  if (!meta?.title || meta.title === "Nueva conversación") {
    setTitleFromFirstQuery(activeId, message);
  } else {
    upsertConversationMeta({ id: activeId });
  }
  renderConversationsPanel();

  streaming = true;
  sendBtn.disabled = true;
  newBtn.disabled = true;
  if (voiceBtn) voiceBtn.disabled = true;
  if (modelSelectEl) modelSelectEl.disabled = true;
  inputEl.value = "";
  setStatus(`Generando con ${selectedBackend() === "neo4j" ? "Neo4j" : "PostgreSQL"}…`);

  appendMessage("user", message);
  const { article, body: assistantBody } = appendMessage("assistant", "");
  article.classList.add("typing");
  let rawAssistant = "";
  let sawThinking = false;
  let sawContent = false;

  try {
    await streamChat({
      message,
      conversationId: activeId,
      retrievalBackend: selectedBackend(),
      model: selectedModel(),
      handlers: {
        onMeta(data) {
          if (data.conversation_id) {
            conversationId = data.conversation_id;
            setActiveConversationId(conversationId);
          }
          if (Array.isArray(data.sources)) {
            upsertReferences(data.sources);
          }
        },
        onThinking(data) {
          sawThinking = true;
          setStatus("Pensando… (el modelo puede tardar un rato)");
          appendThinking(article, data.content || "");
        },
        onToken(data) {
          if (!sawContent && sawThinking) {
            finishThinking(article);
            setStatus("Generando…");
          }
          sawContent = true;
          rawAssistant += data.content || "";
          updateAssistantBody(assistantBody, rawAssistant);
        },
        onDone() {
          if (sawThinking) finishThinking(article);
          updateAssistantBody(assistantBody, rawAssistant);
          upsertConversationMeta({ id: activeId });
          renderConversationsPanel();
          setStatus("");
        },
        onError(data) {
          const text = data.error || "Error en el stream";
          if (!rawAssistant.trim()) {
            article.className = "msg error";
            article.querySelector(".role").textContent = "Error";
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
      article.className = "msg error";
      article.querySelector(".role").textContent = "Error";
      assistantBody.textContent = text;
    } else {
      appendMessage("assistant", text, { error: true });
    }
    setStatus(text, true);
  } finally {
    article.classList.remove("typing");
    streaming = false;
    sendBtn.disabled = false;
    newBtn.disabled = false;
    if (modelSelectEl) modelSelectEl.disabled = false;
    voiceControls?.refresh?.();
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
renderConversationsPanel();
void loadModelSelector();

voiceControls = initVoiceInput({
  formEl,
  inputEl,
  voiceBtn,
  autoSendEl: voiceAutoSendEl,
  getStreaming: () => streaming,
  setStatus,
});

if (conversationId) {
  void switchConversation(conversationId);
} else {
  startNewConversation();
}
