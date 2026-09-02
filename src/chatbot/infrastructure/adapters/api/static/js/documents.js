import {
  deleteDocument,
  listDocumentChunks,
  listDocuments,
  replaceDocument,
  uploadDocument,
} from "./api.js";

const ALLOWED_EXT = new Set([".pdf", ".docx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"]);

const tbody = document.getElementById("docs-body");
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const uploadBtn = document.getElementById("upload-btn");
const chunksPanel = document.getElementById("chunks-panel");
const chunksTitle = document.getElementById("chunks-title");
const chunksBody = document.getElementById("chunks-body");

function setStatus(text, isError = false) {
  statusEl.textContent = text || "";
  statusEl.classList.toggle("error", Boolean(isError));
}

function extensionOf(name) {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function renderRows(documents) {
  tbody.innerHTML = "";
  if (!documents.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6" style="color:var(--ink-muted)">No hay documentos indexados.</td>`;
    tbody.appendChild(tr);
    return;
  }

  for (const doc of documents) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(doc.filename)}</td>
      <td><span class="meta-pill">${escapeHtml(doc.purpose === "notes" ? "Apuntes" : "General")}</span></td>
      <td><span class="meta-pill">${escapeHtml(doc.format)}</span></td>
      <td>${doc.chunk_count}</td>
      <td>${escapeHtml(formatDate(doc.created_at))}</td>
      <td class="actions"></td>
    `;
    const actions = tr.querySelector(".actions");

    const chunksBtn = document.createElement("button");
    chunksBtn.type = "button";
    chunksBtn.className = "secondary compact";
    chunksBtn.textContent = "Chunks";
    chunksBtn.addEventListener("click", () => onShowChunks(doc.id, doc.filename));

    const replaceLabel = document.createElement("label");
    replaceLabel.className = "replace-upload";
    replaceLabel.textContent = "Reemplazar";
    const replaceInput = document.createElement("input");
    replaceInput.type = "file";
    replaceInput.accept = fileInput?.accept || "";
    replaceInput.hidden = true;
    replaceInput.addEventListener("change", () => {
      const file = replaceInput.files?.[0];
      replaceInput.value = "";
      if (file) void onReplace(doc.id, file);
    });
    replaceLabel.appendChild(replaceInput);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "danger";
    btn.textContent = "Borrar";
    btn.addEventListener("click", () => onDelete(doc.id, doc.filename));

    actions.append(chunksBtn, replaceLabel, btn);
    tbody.appendChild(tr);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refresh() {
  setStatus("Cargando…");
  try {
    const data = await listDocuments();
    renderRows(data.documents || []);
    setStatus("");
  } catch (err) {
    setStatus(err.message || "No se pudo listar documentos", true);
  }
}

async function onDelete(id, filename) {
  if (!confirm(`¿Eliminar «${filename}» del índice?`)) return;
  try {
    await deleteDocument(id);
    setStatus(`Eliminado: ${filename}`);
    if (chunksPanel) chunksPanel.hidden = true;
    await refresh();
  } catch (err) {
    setStatus(err.message || "No se pudo eliminar", true);
  }
}

async function onReplace(id, file) {
  const ext = extensionOf(file.name);
  if (!ALLOWED_EXT.has(ext)) {
    setStatus("Formato no permitido. Usa PDF, DOCX, XLSX, XLSM, TXT, MD o CSV.", true);
    return;
  }
  setStatus(`Reemplazando con ${file.name}…`);
  try {
    const result = await replaceDocument(id, file);
    setStatus(`Re-ingerido: ${result.filename} (${result.chunk_count} chunks)`);
    await refresh();
  } catch (err) {
    setStatus(err.message || "No se pudo reemplazar", true);
  }
}

async function onShowChunks(id, filename) {
  if (!chunksPanel || !chunksBody || !chunksTitle) return;
  chunksTitle.textContent = `Chunks — ${filename}`;
  chunksBody.textContent = "Cargando…";
  chunksPanel.hidden = false;
  try {
    const data = await listDocumentChunks(id);
    const chunks = data.chunks || [];
    if (!chunks.length) {
      chunksBody.textContent = "Este documento no tiene chunks.";
      return;
    }
    chunksBody.innerHTML = "";
    for (const chunk of chunks) {
      const article = document.createElement("article");
      article.className = "chunk-card";
      const heading = document.createElement("h3");
      heading.textContent = `#${chunk.index} · ${chunk.id}`;
      const pre = document.createElement("pre");
      pre.textContent = chunk.content;
      article.append(heading, pre);
      chunksBody.appendChild(article);
    }
  } catch (err) {
    chunksBody.textContent = err.message || "No se pudieron cargar los chunks";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus("Selecciona un archivo", true);
    return;
  }

  const ext = extensionOf(file.name);
  if (!ALLOWED_EXT.has(ext)) {
    setStatus("Formato no permitido. Usa PDF, DOCX, XLSX, XLSM, TXT, MD o CSV.", true);
    return;
  }

  uploadBtn.disabled = true;
  const purpose = form.querySelector('input[name="purpose"]:checked')?.value || "general";
  setStatus(`Subiendo ${file.name}…`);
  try {
    const result = await uploadDocument(file, { purpose });
    const tipo = purpose === "notes" ? " (Apuntes)" : "";
    setStatus(`Ingerido: ${result.filename}${tipo} (${result.chunk_count} chunks)`);
    fileInput.value = "";
    await refresh();
  } catch (err) {
    setStatus(err.message || "Error al subir", true);
  } finally {
    uploadBtn.disabled = false;
  }
});

refresh();
