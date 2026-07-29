import {
  createEvalSuite,
  deleteEvalSuite,
  getBioasqDatasetStatus,
  getEvalRun,
  getEvalRunSamples,
  importBioasqDataset,
  listEvalRuns,
  listEvalSuites,
  startEvalRun,
} from "./api.js";

const datasetStatusEl = document.getElementById("dataset-status");
const suiteStatusEl = document.getElementById("suite-status");
const runStatusEl = document.getElementById("run-status");
const suitesBody = document.getElementById("suites-body");
const runsBody = document.getElementById("runs-body");
const suiteDialog = document.getElementById("suite-dialog");
const suiteForm = document.getElementById("suite-form");
const runDetailDialog = document.getElementById("run-detail-dialog");
const runDetailContent = document.getElementById("run-detail-content");

let pollTimer = null;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatPct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function suiteModeLabel(config) {
  if (config.ragas) return "RAGAS";
  if (config.generate) return "Generación";
  return "Retrieval";
}

function readSuiteConfigFromForm() {
  const topKRaw = document.getElementById("cfg-top-k").value.trim();
  return {
    limit: Number(document.getElementById("cfg-limit").value),
    distractors: Number(document.getElementById("cfg-distractors").value),
    top_k: topKRaw ? Number(topKRaw) : null,
    seed: Number(document.getElementById("cfg-seed").value),
    generate: document.getElementById("cfg-generate").checked,
    ragas: document.getElementById("cfg-ragas").checked,
    ragas_timeout: 600,
  };
}

async function refreshDatasetStatus() {
  try {
    const status = await getBioasqDatasetStatus();
    if (!status || status.passage_count === 0) {
      datasetStatusEl.textContent =
        "Dataset no importado. Importa BioASQ a Postgres para evitar descargas repetidas de Hugging Face.";
      return;
    }
    const imported = status.imported_at ? formatDate(status.imported_at) : "—";
    datasetStatusEl.textContent = `${status.passage_count.toLocaleString()} pasajes · ${status.qa_count.toLocaleString()} QA · importado ${imported}`;
  } catch (err) {
    datasetStatusEl.textContent = err.message || "No se pudo leer el estado del dataset";
    datasetStatusEl.classList.add("error");
  }
}

function renderSuites(suites) {
  suitesBody.innerHTML = "";
  if (!suites.length) {
    suitesBody.innerHTML =
      '<tr><td colspan="5" style="color:var(--ink-muted)">No hay suites. Crea una para preparar pruebas RAGAS.</td></tr>';
    return;
  }

  for (const suite of suites) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(suite.name)}</td>
      <td>${suite.sample_ids.length || suite.config.limit}</td>
      <td><span class="meta-pill">${escapeHtml(suiteModeLabel(suite.config))}</span></td>
      <td>${escapeHtml(formatDate(suite.created_at))}</td>
      <td class="actions"></td>
    `;
    const actions = tr.querySelector(".actions");

    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.textContent = "Ejecutar";
    runBtn.addEventListener("click", () => onRunSuite(suite.id, suite.name));
    actions.appendChild(runBtn);

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "danger";
    delBtn.textContent = "Borrar";
    delBtn.addEventListener("click", () => onDeleteSuite(suite.id, suite.name));
    actions.appendChild(delBtn);

    suitesBody.appendChild(tr);
  }
}

function renderRuns(runs) {
  runsBody.innerHTML = "";
  if (!runs.length) {
    runsBody.innerHTML =
      '<tr><td colspan="7" style="color:var(--ink-muted)">Sin ejecuciones todavía.</td></tr>';
    return;
  }

  for (const run of runs) {
    const metrics = run.retrieval_metrics || {};
    const ragas = run.ragas_metrics || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(run.name || run.id.slice(0, 8))}</td>
      <td><span class="meta-pill status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></td>
      <td>${formatPct(metrics.hit_at_k)}</td>
      <td>${formatPct(metrics.recall_at_k)}</td>
      <td>${metrics.mrr != null ? Number(metrics.mrr).toFixed(3) : "—"}</td>
      <td>${ragas.faithfulness != null ? formatPct(ragas.faithfulness) : "—"}</td>
      <td class="actions"></td>
    `;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "secondary compact";
    btn.textContent = "Detalle";
    btn.addEventListener("click", () => showRunDetail(run.id));
    tr.querySelector(".actions").appendChild(btn);
    runsBody.appendChild(tr);
  }
}

async function refreshSuites() {
  try {
    const data = await listEvalSuites();
    renderSuites(data.suites || []);
  } catch (err) {
    suiteStatusEl.textContent = err.message || "Error al listar suites";
    suiteStatusEl.classList.add("error");
  }
}

async function refreshRuns() {
  try {
    const data = await listEvalRuns();
    renderRuns(data.runs || []);
    const hasRunning = (data.runs || []).some((r) => r.status === "running" || r.status === "pending");
    if (hasRunning && !pollTimer) {
      pollTimer = setInterval(refreshRuns, 5000);
    } else if (!hasRunning && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  } catch (err) {
    runStatusEl.textContent = err.message || "Error al listar ejecuciones";
    runStatusEl.classList.add("error");
  }
}

async function onImport(force = false) {
  datasetStatusEl.textContent = force ? "Reimportando dataset…" : "Importando dataset (puede tardar)…";
  try {
    const status = await importBioasqDataset(force);
    datasetStatusEl.textContent = `Importado: ${status.passage_count.toLocaleString()} pasajes, ${status.qa_count.toLocaleString()} QA`;
    await refreshSuites();
  } catch (err) {
    datasetStatusEl.textContent = err.message || "Error al importar";
    datasetStatusEl.classList.add("error");
  }
}

function openSuiteDialog() {
  document.getElementById("suite-name").value = "";
  document.getElementById("suite-description").value = "";
  document.getElementById("cfg-limit").value = "20";
  document.getElementById("cfg-distractors").value = "50";
  document.getElementById("cfg-top-k").value = "";
  document.getElementById("cfg-seed").value = "42";
  document.getElementById("cfg-generate").checked = false;
  document.getElementById("cfg-ragas").checked = false;
  suiteDialog.showModal();
}

async function onSaveSuite(event) {
  event.preventDefault();
  const name = document.getElementById("suite-name").value.trim();
  if (!name) return;

  suiteStatusEl.textContent = "Guardando suite…";
  try {
    await createEvalSuite({
      name,
      description: document.getElementById("suite-description").value.trim() || null,
      config: readSuiteConfigFromForm(),
    });
    suiteDialog.close();
    suiteStatusEl.textContent = `Suite «${name}» creada`;
    await refreshSuites();
  } catch (err) {
    suiteStatusEl.textContent = err.message || "No se pudo crear la suite";
    suiteStatusEl.classList.add("error");
  }
}

async function onDeleteSuite(id, name) {
  if (!confirm(`¿Eliminar la suite «${name}»?`)) return;
  try {
    await deleteEvalSuite(id);
    suiteStatusEl.textContent = `Suite «${name}» eliminada`;
    await refreshSuites();
  } catch (err) {
    suiteStatusEl.textContent = err.message || "No se pudo eliminar";
    suiteStatusEl.classList.add("error");
  }
}

async function onRunSuite(suiteId, name) {
  runStatusEl.textContent = `Iniciando evaluación «${name}»…`;
  try {
    const run = await startEvalRun({ suite_id: suiteId, use_db: true });
    runStatusEl.textContent = `Run ${run.id.slice(0, 8)} en cola (${run.status})`;
    await refreshRuns();
  } catch (err) {
    runStatusEl.textContent = err.message || "No se pudo iniciar la evaluación";
    runStatusEl.classList.add("error");
  }
}

async function showRunDetail(runId) {
  runDetailContent.innerHTML = "<p>Cargando…</p>";
  runDetailDialog.showModal();
  try {
    const [run, samplesData] = await Promise.all([
      getEvalRun(runId),
      getEvalRunSamples(runId, { limit: 20 }),
    ]);
    const metrics = run.retrieval_metrics || {};
    const ragas = run.ragas_metrics || {};
    const sampleRows = (samplesData.samples || [])
      .map(
        (s) => `
        <tr>
          <td>${s.sample_id}</td>
          <td>${escapeHtml(s.question.slice(0, 80))}${s.question.length > 80 ? "…" : ""}</td>
          <td>${s.retrieved_passage_ids.join(", ") || "—"}</td>
        </tr>`
      )
      .join("");

    runDetailContent.innerHTML = `
      <h3>${escapeHtml(run.name || run.id)}</h3>
      <p><strong>Estado:</strong> ${escapeHtml(run.status)} · <strong>Modo:</strong> ${escapeHtml(run.mode)}</p>
      ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
      <div class="metrics-grid">
        <div><span>Hit@k</span><strong>${formatPct(metrics.hit_at_k)}</strong></div>
        <div><span>Recall@k</span><strong>${formatPct(metrics.recall_at_k)}</strong></div>
        <div><span>MRR</span><strong>${metrics.mrr != null ? Number(metrics.mrr).toFixed(3) : "—"}</strong></div>
        <div><span>Faithfulness</span><strong>${formatPct(ragas.faithfulness)}</strong></div>
        <div><span>Answer relevancy</span><strong>${formatPct(ragas.answer_relevancy)}</strong></div>
        <div><span>Context precision</span><strong>${formatPct(ragas.context_precision)}</strong></div>
      </div>
      <h4>Muestras (${samplesData.total})</h4>
      <table class="doc-table">
        <thead><tr><th>ID</th><th>Pregunta</th><th>Pasajes recuperados</th></tr></thead>
        <tbody>${sampleRows || '<tr><td colspan="3">Sin muestras</td></tr>'}</tbody>
      </table>
    `;
  } catch (err) {
    runDetailContent.innerHTML = `<p class="error">${escapeHtml(err.message || "Error")}</p>`;
  }
}

document.getElementById("import-btn").addEventListener("click", () => onImport(false));
document.getElementById("reimport-btn").addEventListener("click", () => onImport(true));
document.getElementById("new-suite-btn").addEventListener("click", openSuiteDialog);
document.getElementById("refresh-runs-btn").addEventListener("click", refreshRuns);
document.getElementById("suite-cancel-btn").addEventListener("click", () => suiteDialog.close());
document.getElementById("run-detail-close").addEventListener("click", () => runDetailDialog.close());
suiteForm.addEventListener("submit", onSaveSuite);

refreshDatasetStatus();
refreshSuites();
refreshRuns();
