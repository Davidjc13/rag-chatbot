/** Índice local de conversaciones (el id lo genera el frontend). */

const INDEX_KEY = "rag_chat_conversations";
const ACTIVE_KEY = "rag_chat_conversation_id";

/**
 * @typedef {{ id: string, title: string, updatedAt: string }} ConversationMeta
 */

export function createConversationId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** @returns {ConversationMeta[]} */
export function loadConversationIndex() {
  try {
    const raw = localStorage.getItem(INDEX_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/** @param {ConversationMeta[]} items */
export function saveConversationIndex(items) {
  localStorage.setItem(INDEX_KEY, JSON.stringify(items));
}

export function getActiveConversationId() {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveConversationId(id) {
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

/**
 * Crea o actualiza metadatos y mueve la conversación al inicio.
 * @param {{ id: string, title?: string }} input
 * @returns {ConversationMeta}
 */
export function upsertConversationMeta({ id, title }) {
  const items = loadConversationIndex().filter((item) => item.id !== id);
  const previous = loadConversationIndex().find((item) => item.id === id);
  const meta = {
    id,
    title: previous?.title || title || "Nueva conversación",
    updatedAt: new Date().toISOString(),
  };
  if (
    title &&
    (!previous?.title || previous.title === "Nueva conversación")
  ) {
    meta.title = title;
  }
  items.unshift(meta);
  saveConversationIndex(items);
  return meta;
}

/** Fija el título con la 1ª query si aún no hay título real. */
export function setTitleFromFirstQuery(id, query) {
  const trimmed = truncateTitle(query);
  return upsertConversationMeta({
    id,
    title: trimmed || "Nueva conversación",
  });
}

export function removeConversation(id) {
  saveConversationIndex(
    loadConversationIndex().filter((entry) => entry.id !== id),
  );
  if (getActiveConversationId() === id) {
    setActiveConversationId(null);
  }
}

export function truncateTitle(text, max = 64) {
  const value = (text || "").trim().replace(/\s+/g, " ");
  if (!value) return "";
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}
