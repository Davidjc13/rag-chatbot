import {
  clearEvalRuns,
  compareEvalRuns,
  createEvalSuite,
  deleteEvalRun,
  deleteEvalSuite,
  getBioasqDatasetStatus,
  getEvalRun,
  getEvalRunSamples,
  importBioasqDataset,
  importJsonDataset,
  listEvalDatasets,
  listEvalRuns,
  listEvalSuites,
  startAbTest,
  startEvalRun,
} from "./api.js";

const datasetStatusEl = document.getElementById("dataset-status");
const suiteStatusEl = document.getElementById("suite-status");
const runStatusEl = document.getElementById("run-status");
const abStatusEl = document.getElementById("ab-status");
const datasetsBody = document.getElementById("datasets-body");
const suitesBody = document.getElementById("suites-body");
const runsBody = document.getElementById("runs-body");
const suiteDialog = document.getElementById("suite-dialog");
const suiteForm = document.getElementById("suite-form");
const abDialog = document.getElementById("ab-dialog");
const abForm = document.getElementById("ab-form");
const runDetailDialog = document.getElementById("run-detail-dialog");
const runDetailContent = document.getElementById("run-detail-content");
const compareResultsEl = document.getElementById("compare-results");
const compareRunA = document.getElementById("compare-run-a");
const compareRunB = document.getElementById("compare-run-b");
const suiteDatasetSelect = document.getElementById("suite-dataset-id");
const abSuiteSelect = document.getElementById("ab-suite-id");

let pollTimer = null;
let cachedRuns = [];

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

function formatDeepevalMetric(deepeval) {
  if (!deepeval || typeof deepeval !== "object") return "—";
  if (deepeval._error) return "error";
  const preferred = [
    deepeval.answer_relevancy,
    deepeval["Answer Relevancy"],
    deepeval.faithfulness,
    deepeval["Faithfulness"],
    deepeval.contextual_relevancy,
    deepeval["Contextual Relevancy"],
  ];
  for (const value of preferred) {
    if (value != null && !Number.isNaN(Number(value))) {
      return formatPct(value);
    }
  }
  const scores = Object.entries(deepeval)
    .filter(([key, value]) => !key.startsWith("_") && typeof value === "number")
    .map(([, value]) => Number(value));
  if (!scores.length) return "—";
  const avg = scores.reduce((sum, value) => sum + value, 0) / scores.length;
  return formatPct(avg);
}

function deepevalDetailLines(deepeval) {
  if (!deepeval || typeof deepeval !== "object") return [];
  return Object.entries(deepeval)
    .filter(([key, value]) => !key.startsWith("_") && typeof value === "number")
    .map(([key, value]) => `${key}: ${formatPct(value)}`);
}

function formatDelta(value) {
  if (value == null || Number.isNaN(value)) return "—";
  const pct = Number(value) * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)} pp`;
}

function truncateText(value, max = 120) {
  const text = String(value || "").trim();
  if (!text) return "—";
  if (text.length <= max) return text;
  return `${text.slice(0, max)}…`;
}

function renderAnswerBlock(label, value) {
  const text = String(value || "").trim();
  if (!text) {
    return `<div class="sample-field"><span>${escapeHtml(label)}</span><em class="muted">Sin respuesta</em></div>`;
  }
  return `
    <div class="sample-field">
      <span>${escapeHtml(label)}</span>
      <div class="sample-answer">${escapeHtml(text)}</div>
    </div>`;
}

function renderSampleRow(sample, { showAnswers = true } = {}) {
  const answerCell = showAnswers
    ? `<td class="sample-answer-cell">${escapeHtml(truncateText(sample.answer, 160))}</td>`
    : "";
  const groundTruthCell = showAnswers
    ? `<td class="sample-answer-cell">${escapeHtml(truncateText(sample.ground_truth, 100))}</td>`
    : "";
  return `
    <tr>
      <td>${sample.sample_id}</td>
      <td>${escapeHtml(truncateText(sample.question, 100))}</td>
      ${groundTruthCell}
      ${answerCell}
      <td>${sample.retrieved_passage_ids.join(", ") || "—"}</td>
    </tr>`;
}

function renderSampleCards(samples, { showAnswers = true } = {}) {
  if (!samples.length) {
    return '<p class="muted">Sin muestras guardadas.</p>';
  }
  return samples
    .map(
      (sample) => `
      <article class="sample-card">
        <header>
          <strong>#${sample.sample_id}</strong>
          <span>${escapeHtml(truncateText(sample.question, 140))}</span>
        </header>
        ${showAnswers ? renderAnswerBlock("Ground truth", sample.ground_truth) : ""}
        ${showAnswers ? renderAnswerBlock("Respuesta generada", sample.answer) : ""}
        <div class="sample-field">
          <span>Pasajes recuperados</span>
          <div>${escapeHtml(sample.retrieved_passage_ids.join(", ") || "—")}</div>
        </div>
      </article>`,
    )
    .join("");
}

function suiteModeLabel(config) {
  const parts = [];
  if (config.generate) parts.push("Generación");
  else parts.push("Retrieval");
  if (config.ragas) parts.push("RAGAS");
  if (config.deepeval) parts.push("DeepEval");
  return parts.join(" + ");
}

function readSuiteConfigFromForm() {
  const topKRaw = document.getElementById("cfg-top-k").value.trim();
  const llmModel = document.getElementById("cfg-llm-model").value.trim();
  return {
    limit: Number(document.getElementById("cfg-limit").value),
    distractors: Number(document.getElementById("cfg-distractors").value),
    top_k: topKRaw ? Number(topKRaw) : null,
    seed: Number(document.getElementById("cfg-seed").value),
    generate: document.getElementById("cfg-generate").checked,
    ragas: document.getElementById("cfg-ragas").checked,
    deepeval: document.getElementById("cfg-deepeval").checked,
    ragas_timeout: 600,
    deepeval_timeout: 600,
    llm_model: llmModel || null,
  };
}

function buildVariantConfig(name, model, flags) {
  const config = {
    generate: flags.generate,
    ragas: flags.ragas,
    deepeval: flags.deepeval,
    ragas_timeout: 600,
    deepeval_timeout: 600,
  };
  if (model) config.llm_model = model;
  return { name, config };
}

function populateDatasetSelects(datasets) {
  const options = (datasets || [])
    .map((d) => `<option value="${escapeHtml(d.dataset_id)}">${escapeHtml(d.name)} (${escapeHtml(d.dataset_id)})</option>`)
    .join("");
  const fallback = '<option value="bioasq">BioASQ (bioasq)</option>';
  suiteDatasetSelect.innerHTML = options || fallback;
}

function populateRunCompareSelects(runs) {
  const completed = (runs || []).filter((r) => r.status === "completed");
  const options = completed
    .map((r) => {
      const label = `${r.variant_label ? `[${r.variant_label}] ` : ""}${r.name || r.id.slice(0, 8)}`;
      return `<option value="${escapeHtml(r.id)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  compareRunA.innerHTML = `<option value="">Selecciona run A</option>${options}`;
  compareRunB.innerHTML = `<option value="">Selecciona run B</option>${options}`;
}

async function refreshDatasets() {
  try {
    const data = await listEvalDatasets();
    const datasets = data.datasets || [];
    renderDatasets(datasets);
    populateDatasetSelects(datasets);
    if (!datasets.length) {
      const bioasq = await getBioasqDatasetStatus();
      if (bioasq?.passage_count) {
        datasetStatusEl.textContent = "Solo BioASQ disponible en memoria de estado.";
      } else {
        datasetStatusEl.textContent =
          "No hay datasets importados. Importa BioASQ o sube un JSON personalizado.";
      }
      return;
    }
    datasetStatusEl.textContent = `${datasets.length} dataset(s) importado(s) en Postgres.`;
  } catch (err) {
    datasetStatusEl.textContent = err.message || "No se pudieron listar datasets";
    datasetStatusEl.classList.add("error");
  }
}

function renderDatasets(datasets) {
  datasetsBody.innerHTML = "";
  if (!datasets.length) {
    datasetsBody.innerHTML =
      '<tr><td colspan="5" style="color:var(--ink-muted)">Sin datasets importados.</td></tr>';
    return;
  }
  for (const item of datasets) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(item.dataset_id)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.hf_source)}</td>
      <td>${Number(item.passage_count).toLocaleString()}</td>
      <td>${Number(item.qa_count).toLocaleString()}</td>
    `;
    datasetsBody.appendChild(tr);
  }
}

function renderSuites(suites) {
  suitesBody.innerHTML = "";
  abSuiteSelect.innerHTML = "";
  if (!suites.length) {
    suitesBody.innerHTML =
      '<tr><td colspan="6" style="color:var(--ink-muted)">No hay suites. Crea una para preparar evaluaciones.</td></tr>';
    return;
  }

  for (const suite of suites) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(suite.name)}</td>
      <td>${escapeHtml(suite.dataset_id)}</td>
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

    const opt = document.createElement("option");
    opt.value = suite.id;
    opt.textContent = `${suite.name} (${suite.dataset_id})`;
    abSuiteSelect.appendChild(opt);
  }
}

function renderRuns(runs) {
  cachedRuns = runs || [];
  runsBody.innerHTML = "";
  populateRunCompareSelects(cachedRuns);
  if (!runs.length) {
    runsBody.innerHTML =
      '<tr><td colspan="9" style="color:var(--ink-muted)">Sin ejecuciones todavía.</td></tr>';
    return;
  }

  for (const run of runs) {
    const metrics = run.retrieval_metrics || {};
    const ragas = run.ragas_metrics || {};
    const deepeval = run.deepeval_metrics || {};
    const faith = ragas.faithfulness ?? ragas.Faithfulness;
    const relevancy = formatDeepevalMetric(deepeval);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(run.name || run.id.slice(0, 8))}</td>
      <td><span class="meta-pill status-${escapeHtml(run.status)}">${escapeHtml(run.status)}</span></td>
      <td>${escapeHtml(run.variant_label || "—")}</td>
      <td>${formatPct(metrics.hit_at_k)}</td>
      <td>${formatPct(metrics.recall_at_k)}</td>
      <td>${metrics.mrr != null ? Number(metrics.mrr).toFixed(3) : "—"}</td>
      <td>${faith != null ? formatPct(faith) : "—"}</td>
      <td class="${deepeval._error ? "run-error-hint" : ""}">${relevancy === "error" ? escapeHtml(String(deepeval._error || "error").slice(0, 40)) : relevancy}</td>
      <td class="actions"></td>
    `;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "secondary compact";
    btn.textContent = "Detalle";
    btn.addEventListener("click", () => showRunDetail(run.id));
    tr.querySelector(".actions").appendChild(btn);

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "danger compact";
    delBtn.textContent = "Borrar";
    delBtn.addEventListener("click", () => onDeleteRun(run.id, run.name || run.id.slice(0, 8)));
    tr.querySelector(".actions").appendChild(delBtn);

    runsBody.appendChild(tr);
    if (run.status === "failed" && run.error) {
      const errTr = document.createElement("tr");
      errTr.className = "run-error-row";
      errTr.innerHTML = `<td colspan="9" class="run-error-hint">${escapeHtml(run.error.split("\n")[0].slice(0, 200))}</td>`;
      runsBody.appendChild(errTr);
    }
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

async function onImportBioasq(force = false) {
  datasetStatusEl.textContent = force ? "Reimportando BioASQ…" : "Importando BioASQ…";
  try {
    await importBioasqDataset(force);
    datasetStatusEl.textContent = "BioASQ importado correctamente";
    await refreshDatasets();
    await refreshSuites();
  } catch (err) {
    datasetStatusEl.textContent = err.message || "Error al importar BioASQ";
    datasetStatusEl.classList.add("error");
  }
}

async function onImportJson() {
  const fileInput = document.getElementById("json-file");
  const file = fileInput.files?.[0];
  if (!file) {
    datasetStatusEl.textContent = "Selecciona un archivo JSON";
    return;
  }
  const datasetId = document.getElementById("json-dataset-id").value.trim() || null;
  datasetStatusEl.textContent = "Importando JSON…";
  try {
    const status = await importJsonDataset(file, { datasetId });
    datasetStatusEl.textContent = `JSON importado: ${status.name} (${status.dataset_id})`;
    fileInput.value = "";
    await refreshDatasets();
  } catch (err) {
    datasetStatusEl.textContent = err.message || "Error al importar JSON";
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
  document.getElementById("cfg-deepeval").checked = false;
  document.getElementById("cfg-llm-model").value = "";
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
      dataset_id: suiteDatasetSelect.value || "bioasq",
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

async function onDeleteRun(runId, label) {
  if (!confirm(`¿Eliminar la ejecución «${label}»?`)) return;
  try {
    await deleteEvalRun(runId);
    runStatusEl.textContent = `Ejecución «${label}» eliminada`;
    compareResultsEl.hidden = true;
    await refreshRuns();
  } catch (err) {
    runStatusEl.textContent = err.message || "No se pudo eliminar la ejecución";
    runStatusEl.classList.add("error");
  }
}

async function onClearRuns() {
  if (!cachedRuns.length) {
    runStatusEl.textContent = "No hay ejecuciones que limpiar";
    return;
  }
  if (
    !confirm(
      "¿Eliminar todas las ejecuciones y experimentos A/B?\n\nLas suites y datasets no se borran.",
    )
  ) {
    return;
  }
  runStatusEl.textContent = "Limpiando historial…";
  try {
    const result = await clearEvalRuns();
    runStatusEl.textContent = `Eliminadas ${result.runs_deleted} ejecución(es) y ${result.experiments_deleted} experimento(s)`;
    abStatusEl.textContent = "";
    compareResultsEl.hidden = true;
    compareResultsEl.innerHTML = "";
    await refreshRuns();
  } catch (err) {
    runStatusEl.textContent = err.message || "No se pudo limpiar el historial";
    runStatusEl.classList.add("error");
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

function openAbDialog() {
  abForm.reset();
  document.getElementById("ab-a-name").value = "Variante A";
  document.getElementById("ab-b-name").value = "Variante B";
  document.getElementById("ab-generate").checked = true;
  abDialog.showModal();
}

async function onStartAbTest(event) {
  event.preventDefault();
  const flags = {
    generate: document.getElementById("ab-generate").checked,
    ragas: document.getElementById("ab-ragas").checked,
    deepeval: document.getElementById("ab-deepeval").checked,
  };
  abStatusEl.textContent = "Lanzando experimento A/B…";
  try {
    const experiment = await startAbTest({
      suite_id: document.getElementById("ab-suite-id").value,
      name: document.getElementById("ab-name").value.trim() || null,
      variant_a: buildVariantConfig(
        document.getElementById("ab-a-name").value.trim(),
        document.getElementById("ab-a-model").value.trim(),
        flags,
      ),
      variant_b: buildVariantConfig(
        document.getElementById("ab-b-name").value.trim(),
        document.getElementById("ab-b-model").value.trim(),
        flags,
      ),
    });
    abDialog.close();
    abStatusEl.textContent = `Experimento ${experiment.id.slice(0, 8)} iniciado (runs A/B en cola)`;
    await refreshRuns();
  } catch (err) {
    abStatusEl.textContent = err.message || "No se pudo lanzar el A/B";
    abStatusEl.classList.add("error");
  }
}

async function onCompareRuns() {
  const runA = compareRunA.value;
  const runB = compareRunB.value;
  if (!runA || !runB) {
    abStatusEl.textContent = "Selecciona dos runs completados";
    return;
  }
  if (runA === runB) {
    abStatusEl.textContent = "Selecciona dos runs distintos";
    return;
  }
  compareResultsEl.hidden = false;
  compareResultsEl.innerHTML = "<p>Calculando comparación…</p>";
  try {
    const result = await compareEvalRuns(runA, runB);
    renderComparison(result);
    abStatusEl.textContent = "Comparación lista";
  } catch (err) {
    compareResultsEl.innerHTML = `<p class="error">${escapeHtml(err.message || "Error")}</p>`;
  }
}

function renderComparison(result) {
  const retrievalRows = Object.entries(result.retrieval_delta || {})
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${formatDelta(value)}</td></tr>`)
    .join("");
  const deepevalRows = Object.entries(result.deepeval_delta || {})
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${formatDelta(value)}</td></tr>`)
    .join("");
  const sampleRows = (result.samples || [])
    .slice(0, 15)
    .map(
      (s) => `
      <tr>
        <td>${s.sample_id}</td>
        <td>${escapeHtml(truncateText(s.question, 60))}</td>
        <td class="sample-answer-cell">${escapeHtml(truncateText(s.answer_a, 120))}</td>
        <td class="sample-answer-cell">${escapeHtml(truncateText(s.answer_b, 120))}</td>
      </tr>`,
    )
    .join("");

  compareResultsEl.innerHTML = `
    <h3>${escapeHtml(result.run_a_name || result.run_a_id)} vs ${escapeHtml(result.run_b_name || result.run_b_id)}</h3>
    <div class="metrics-grid">
      <div><span>Win rate B (score)</span><strong>${formatPct(result.win_rates?.retrieval_score_b)}</strong></div>
    </div>
    <h4>Δ Retrieval (B − A)</h4>
    <table class="doc-table compact"><tbody>${retrievalRows || "<tr><td colspan=2>Sin delta</td></tr>"}</tbody></table>
    <h4>Δ DeepEval (B − A)</h4>
    <table class="doc-table compact"><tbody>${deepevalRows || "<tr><td colspan=2>Sin delta</td></tr>"}</tbody></table>
    <h4>Muestras comparadas</h4>
    <table class="doc-table compact">
      <thead><tr><th>ID</th><th>Pregunta</th><th>Resp. A</th><th>Resp. B</th></tr></thead>
      <tbody>${sampleRows || "<tr><td colspan=4>Sin muestras alineadas</td></tr>"}</tbody>
    </table>
  `;
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
    const deepeval = run.deepeval_metrics || {};
    const deepevalLines = deepevalDetailLines(deepeval);
    const showAnswers = Boolean(run.config?.generate);
    const sampleRows = (samplesData.samples || [])
      .map((s) => renderSampleRow(s, { showAnswers }))
      .join("");
    const sampleCards = renderSampleCards(samplesData.samples || [], { showAnswers });
    const tableHead = showAnswers
      ? "<tr><th>ID</th><th>Pregunta</th><th>Ground truth</th><th>Respuesta generada</th><th>Pasajes</th></tr>"
      : "<tr><th>ID</th><th>Pregunta</th><th>Pasajes recuperados</th></tr>";

    runDetailContent.innerHTML = `
      <h3>${escapeHtml(run.name || run.id)}</h3>
      <p><strong>Estado:</strong> ${escapeHtml(run.status)} · <strong>Modo:</strong> ${escapeHtml(run.mode)}</p>
      ${run.variant_label ? `<p><strong>Variante:</strong> ${escapeHtml(run.variant_label)}</p>` : ""}
      ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
      <div class="metrics-grid">
        <div><span>Hit@k</span><strong>${formatPct(metrics.hit_at_k)}</strong></div>
        <div><span>Recall@k</span><strong>${formatPct(metrics.recall_at_k)}</strong></div>
        <div><span>MRR</span><strong>${metrics.mrr != null ? Number(metrics.mrr).toFixed(3) : "—"}</strong></div>
        <div><span>Faithfulness</span><strong>${formatPct(ragas.faithfulness)}</strong></div>
        <div><span>Answer relevancy (RAGAS)</span><strong>${formatPct(ragas.answer_relevancy)}</strong></div>
        <div><span>DeepEval (media)</span><strong>${formatDeepevalMetric(deepeval)}</strong></div>
      </div>
      ${deepeval._error ? `<p class="error">${escapeHtml(String(deepeval._error))}</p>` : ""}
      ${deepevalLines.length ? `<p><strong>DeepEval:</strong> ${deepevalLines.map(escapeHtml).join(" · ")}</p>` : ""}
      <h4>Muestras (${samplesData.total})</h4>
      ${showAnswers ? `<div class="sample-cards">${sampleCards}</div>` : ""}
      <table class="doc-table compact sample-table">
        <thead>${tableHead}</thead>
        <tbody>${sampleRows || `<tr><td colspan="${showAnswers ? 5 : 3}">Sin muestras</td></tr>`}</tbody>
      </table>
    `;
  } catch (err) {
    runDetailContent.innerHTML = `<p class="error">${escapeHtml(err.message || "Error")}</p>`;
  }
}

document.getElementById("import-btn").addEventListener("click", () => onImportBioasq(false));
document.getElementById("reimport-btn").addEventListener("click", () => onImportBioasq(true));
document.getElementById("json-import-btn").addEventListener("click", onImportJson);
document.getElementById("new-suite-btn").addEventListener("click", openSuiteDialog);
document.getElementById("open-ab-btn").addEventListener("click", openAbDialog);
document.getElementById("compare-btn").addEventListener("click", onCompareRuns);
document.getElementById("refresh-runs-btn").addEventListener("click", refreshRuns);
document.getElementById("clear-runs-btn").addEventListener("click", onClearRuns);
document.getElementById("clear-experiments-btn").addEventListener("click", onClearRuns);
document.getElementById("suite-cancel-btn").addEventListener("click", () => suiteDialog.close());
document.getElementById("ab-cancel-btn").addEventListener("click", () => abDialog.close());
document.getElementById("run-detail-close").addEventListener("click", () => runDetailDialog.close());
suiteForm.addEventListener("submit", onSaveSuite);
abForm.addEventListener("submit", onStartAbTest);

refreshDatasets();
refreshSuites();
refreshRuns();
