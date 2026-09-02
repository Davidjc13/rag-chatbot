/** Índice de conversaciones: el servidor es la fuente de verdad; localStorage solo guarda el id activo. */

const ACTIVE_KEY = "rag_chat_conversation_id";

/**
 * @typedef {{ id: string, title: string, updatedAt: string, preview?: string }} ConversationMeta
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

export function getActiveConversationId() {
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveConversationId(id) {
  if (id) localStorage.setItem(ACTIVE_KEY, id);
  else localStorage.removeItem(ACTIVE_KEY);
}

export function truncateTitle(text, max = 64) {
  const value = (text || "").trim().replace(/\s+/g, " ");
  if (!value) return "";
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}
