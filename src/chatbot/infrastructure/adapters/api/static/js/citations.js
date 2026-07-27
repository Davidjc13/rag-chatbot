/** Parseo y render de citas RAG embebidas en respuestas. */

const CITATION_RE =
  /<index\s*=\s*(\d+)\s*,\s*source\s*=\s*([^,>]+?)\s*,\s*title\s*=\s*([^,>]+?)\s*,\s*id\s*=\s*([^>]+?)>/gi;

/**
 * Separa una cita incompleta al final del stream (sin `>` de cierre).
 */
export function splitIncompleteCitation(text) {
  const open = text.lastIndexOf("<index");
  if (open === -1) {
    return { complete: text, pending: "" };
  }
  const tail = text.slice(open);
  if (tail.includes(">")) {
    return { complete: text, pending: "" };
  }
  return { complete: text.slice(0, open), pending: tail };
}

export function extractCitations(text) {
  const refs = [];
  const seenIds = new Set();
  CITATION_RE.lastIndex = 0;
  let match;
  while ((match = CITATION_RE.exec(text)) !== null) {
    const id = match[4].trim();
    if (seenIds.has(id)) continue;
    seenIds.add(id);
    refs.push({
      index: Number(match[1]),
      source: match[2].trim(),
      title: match[3].trim(),
      id,
    });
  }
  return refs;
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/**
 * Convierte texto con citas en HTML seguro.
 * Un mismo documento (id) solo se muestra una vez como superíndice.
 */
export function renderCitedHtml(text) {
  const { complete, pending } = splitIncompleteCitation(text);
  const refs = extractCitations(complete);
  const idToIndex = new Map(refs.map((ref) => [ref.id, ref.index]));
  const renderedIds = new Set();
  const parts = [];
  let last = 0;
  CITATION_RE.lastIndex = 0;
  let match;
  while ((match = CITATION_RE.exec(complete)) !== null) {
    parts.push(escapeHtml(complete.slice(last, match.index)));
    const id = match[4].trim();
    const index = idToIndex.get(id) ?? Number(match[1]);
    if (!renderedIds.has(id)) {
      renderedIds.add(id);
      parts.push(
        `<sup class="cite"><a href="#ref-${index}" data-cite="${index}">${index}</a></sup>`,
      );
    }
    last = match.index + match[0].length;
  }
  parts.push(escapeHtml(complete.slice(last)));
  if (pending && !pending.startsWith("<index")) {
    parts.push(escapeHtml(pending));
  }
  return { html: parts.join(""), refs, pending };
}
