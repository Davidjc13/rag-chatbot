"""Serialización de tablas a Markdown semántico."""

from __future__ import annotations


def _cell(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("|", "\\|").replace("\n", " ").strip()
    return text


def table_to_markdown(
    headers: list[str],
    rows: list[list[object | None]],
    *,
    title: str | None = None,
) -> str:
    """
    Convierte una tabla tabular en Markdown con cabeceras.

    Las cabeceras aportan el contexto semántico que las celdas sueltas no tienen.
    """
    clean_headers = [_cell(h) or f"col_{i + 1}" for i, h in enumerate(headers)]
    if not clean_headers:
        width = max((len(r) for r in rows), default=0)
        clean_headers = [f"col_{i + 1}" for i in range(width)]

    width = len(clean_headers)
    normalized_rows: list[list[str]] = []
    for row in rows:
        cells = [_cell(c) for c in row]
        if len(cells) < width:
            cells.extend([""] * (width - len(cells)))
        elif len(cells) > width:
            cells = cells[:width]
        if any(cells):
            normalized_rows.append(cells)

    lines: list[str] = []
    if title:
        lines.append(f"**{title}**")
        lines.append("")

    lines.append("| " + " | ".join(clean_headers) + " |")
    lines.append("| " + " | ".join("---" for _ in clean_headers) + " |")
    for row in normalized_rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def split_markdown_table_by_rows(
    markdown: str,
    *,
    max_chars: int,
) -> list[str]:
    """
    Fallback: parte una tabla Markdown por filas repitiendo la cabecera.

    Nunca corta una fila a medias.
    """
    lines = [line for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return [markdown]

    # Detectar bloque de tabla (líneas que empiezan por |)
    title_lines: list[str] = []
    table_lines: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            table_lines.append(line)
        elif not table_lines:
            title_lines.append(line)

    if len(table_lines) < 2:
        return [markdown]

    header = table_lines[0]
    separator = table_lines[1]
    data_rows = table_lines[2:]
    prefix = "\n".join(title_lines).strip()

    def wrap(rows: list[str]) -> str:
        body = "\n".join([header, separator, *rows])
        if prefix:
            return f"{prefix}\n\n{body}"
        return body

    if not data_rows:
        return [wrap([])]

    chunks: list[str] = []
    current: list[str] = []
    for row in data_rows:
        candidate = wrap([*current, row])
        if current and len(candidate) > max_chars:
            chunks.append(wrap(current))
            current = [row]
        else:
            current.append(row)
    if current:
        chunks.append(wrap(current))
    return chunks
