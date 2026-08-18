/** Cliente HTTP hacia /api/v1 */

const API_BASE = "/api/v1";

export async function apiJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (response.status === 204) {
    return null;
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.error || `Error HTTP ${response.status}`);
    error.code = data.code;
    error.status = response.status;
    error.detail = data.detail;
    throw error;
  }
  return data;
}

export async function listDocuments() {
  return apiJson("/documents");
}

export async function deleteDocument(documentId) {
  return apiJson(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
}

export async function uploadDocument(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  return apiJson("/documents", { method: "POST", body });
}

export async function transcribeAudio(file) {
  const body = new FormData();
  body.append("file", file, file.name);
  return apiJson("/transcribe", { method: "POST", body });
}

export async function getBioasqDatasetStatus() {
  return apiJson("/evals/datasets/bioasq");
}

export async function importBioasqDataset(force = false) {
  return apiJson("/evals/datasets/bioasq/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
}

export async function listEvalDatasets() {
  return apiJson("/evals/datasets");
}

export async function importJsonDataset(file, { datasetId = null, force = false } = {}) {
  const body = new FormData();
  body.append("file", file, file.name);
  const params = new URLSearchParams();
  if (datasetId) params.set("dataset_id", datasetId);
  if (force) params.set("force", "true");
  const query = params.toString();
  return apiJson(`/evals/datasets/json/import${query ? `?${query}` : ""}`, {
    method: "POST",
    body,
  });
}

export async function listEvalSuites() {
  return apiJson("/evals/suites");
}

export async function createEvalSuite(payload) {
  return apiJson("/evals/suites", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function deleteEvalSuite(suiteId) {
  return apiJson(`/evals/suites/${encodeURIComponent(suiteId)}`, {
    method: "DELETE",
  });
}

export async function listEvalRuns() {
  return apiJson("/evals/runs");
}

export async function startEvalRun(payload) {
  return apiJson("/evals/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getEvalRun(runId) {
  return apiJson(`/evals/runs/${encodeURIComponent(runId)}`);
}

export async function deleteEvalRun(runId) {
  return apiJson(`/evals/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
}

export async function clearEvalRuns() {
  return apiJson("/evals/runs", { method: "DELETE" });
}

export async function getEvalRunSamples(runId, { offset = 0, limit = 50 } = {}) {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  return apiJson(`/evals/runs/${encodeURIComponent(runId)}/samples?${params}`);
}

export async function startAbTest(payload) {
  return apiJson("/evals/ab-test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function compareEvalRuns(runA, runB) {
  const params = new URLSearchParams({ run_a: runA, run_b: runB });
  return apiJson(`/evals/compare?${params}`);
}

export async function getConversation(conversationId) {
  return apiJson(`/conversations/${encodeURIComponent(conversationId)}`);
}

export async function listModels() {
  return apiJson("/models");
}

/**
 * Consume SSE de POST /chat/stream.
 * handlers: { onMeta, onThinking, onToken, onDone, onCancelled, onError }
 */
export async function streamChat({
  message,
  conversationId,
  retrievalBackend,
  model,
  handlers,
  signal,
}) {
  let reader;
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message,
        conversation_id: conversationId || null,
        retrieval_backend: retrievalBackend || "postgres",
        model: model || null,
      }),
      signal,
    });

    if (!response.ok || !response.body) {
      let payload = {};
      try {
        payload = await response.json();
      } catch {
        /* ignore */
      }
      const error = new Error(payload.error || `Error HTTP ${response.status}`);
      error.code = payload.code;
      throw error;
    }

    reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      if (signal?.aborted) {
        await reader.cancel();
        break;
      }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        dispatchSseBlock(raw, handlers);
      }
    }

    if (buffer.trim()) {
      dispatchSseBlock(buffer, handlers);
    }
  } catch (err) {
    if (err.name === "AbortError") {
      if (handlers.onCancelled) handlers.onCancelled();
      return;
    }
    throw err;
  }
}

function dispatchSseBlock(raw, handlers) {
  let eventName = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) return;

  let data;
  try {
    data = JSON.parse(dataLines.join("\n"));
  } catch {
    return;
  }

  if (eventName === "meta" && handlers.onMeta) handlers.onMeta(data);
  else if (eventName === "thinking" && handlers.onThinking) handlers.onThinking(data);
  else if (eventName === "token" && handlers.onToken) handlers.onToken(data);
  else if (eventName === "done" && handlers.onDone) handlers.onDone(data);
  else if (eventName === "cancelled" && handlers.onCancelled) handlers.onCancelled(data);
  else if (eventName === "error" && handlers.onError) handlers.onError(data);
}
