import { deleteDocument, listDocuments, uploadDocument } from "./api.js";

const ALLOWED_EXT = new Set([".pdf", ".docx", ".xlsx", ".xlsm"]);

const tbody = document.getElementById("docs-body");
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const statusEl = document.getElementById("status");
const uploadBtn = document.getElementById("upload-btn");

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
    tr.innerHTML = `<td colspan="5" style="color:var(--ink-muted)">No hay documentos indexados.</td>`;
    tbody.appendChild(tr);
    return;
  }

  for (const doc of documents) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(doc.filename)}</td>
      <td><span class="meta-pill">${escapeHtml(doc.format)}</span></td>
      <td>${doc.chunk_count}</td>
      <td>${escapeHtml(formatDate(doc.created_at))}</td>
      <td class="actions"></td>
    `;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "danger";
    btn.textContent = "Borrar";
    btn.addEventListener("click", () => onDelete(doc.id, doc.filename));
    tr.querySelector(".actions").appendChild(btn);
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
    await refresh();
  } catch (err) {
    setStatus(err.message || "No se pudo eliminar", true);
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
    setStatus("Formato no permitido. Usa PDF, DOCX, XLSX o XLSM.", true);
    return;
  }

  uploadBtn.disabled = true;
  setStatus(`Subiendo ${file.name}…`);
  try {
    const result = await uploadDocument(file);
    setStatus(`Ingerido: ${result.filename} (${result.chunk_count} chunks)`);
    fileInput.value = "";
    await refresh();
  } catch (err) {
    setStatus(err.message || "Error al subir", true);
  } finally {
    uploadBtn.disabled = false;
  }
});

refresh();
