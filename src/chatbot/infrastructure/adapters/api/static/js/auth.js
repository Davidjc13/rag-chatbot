/** API key en localStorage: se envía como Bearer en /api/v1. */

const STORAGE_KEY = "rag_chat_api_key";

export function getApiKey() {
  try {
    return localStorage.getItem(STORAGE_KEY) || "";
  } catch {
    return "";
  }
}

export function setApiKey(value) {
  const trimmed = (value || "").trim();
  try {
    if (trimmed) localStorage.setItem(STORAGE_KEY, trimmed);
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore quota / private mode */
  }
}

export function authHeaders() {
  const key = getApiKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

function mountApiKeyControl() {
  const bar = document.querySelector(".topbar");
  if (!bar || bar.querySelector(".api-key-field")) return;

  const label = document.createElement("label");
  label.className = "api-key-field";
  label.title =
    "Si AUTH_ENABLED=true, pega la API key. Health y estáticos no la requieren.";

  const text = document.createElement("span");
  text.textContent = "API key";

  const input = document.createElement("input");
  input.type = "password";
  input.autocomplete = "off";
  input.placeholder = "Bearer…";
  input.value = getApiKey();
  input.setAttribute("aria-label", "API key");
  input.addEventListener("change", () => setApiKey(input.value));
  input.addEventListener("blur", () => setApiKey(input.value));

  label.append(text, input);
  bar.appendChild(label);
}

mountApiKeyControl();
