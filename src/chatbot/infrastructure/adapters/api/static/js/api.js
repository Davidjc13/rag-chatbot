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

export async function getConversation(conversationId) {
  return apiJson(`/conversations/${encodeURIComponent(conversationId)}`);
}

/**
 * Consume SSE de POST /chat/stream.
 * handlers: { onMeta, onThinking, onToken, onDone, onError }
 */
export async function streamChat({ message, conversationId, handlers }) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
    }),
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

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
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
  else if (eventName === "error" && handlers.onError) handlers.onError(data);
}
