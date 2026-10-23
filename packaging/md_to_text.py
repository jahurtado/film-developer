#!/usr/bin/env python3
"""Render a Markdown file as readable plain text for the installer.

Inno Setup shows InfoAfterFile as plain text -- it does not interpret Markdown --
so a `.md` shown there leaks its syntax: `#` on headings, `**` around bold, and
worst of all a pipe table whose `|---|` separator row and unaligned columns are
unreadable. This strips the markup and lays the table out in aligned columns, so
the same one source (THIRD-PARTY-LICENSES.md, which GitHub still renders) can be
shown in the installer without looking broken. Output is UTF-8 with a BOM,
because the text carries em dashes and Inno reads a BOM'd file as UTF-8.

    python md_to_text.py THIRD-PARTY-LICENSES.md THIRD-PARTY-LICENSES.txt
"""
import re
import sys


def strip_inline(s):
    """Undo the inline markup that would show as literal characters."""
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", s)  # [text](url) -> text (url)
    s = re.sub(r"<((?:https?|mailto:)[^>]+)>", r"\1", s)    # <url> -> url
    s = s.replace("**", "").replace("`", "")                # bold, inline code
    return s


def render_table(rows):
    """A Markdown pipe table as aligned columns; the `|---|` row is dropped."""
    cells = []
    for r in rows:
        parts = [c.strip() for c in r.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in parts):        # separator row
            continue
        cells.append([strip_inline(c) for c in parts])
    if not cells:
        return []
    width = [max(len(row[i]) for row in cells) for i in range(len(cells[0]))]
    return ["  ".join(c.ljust(width[i]) for i, c in enumerate(row)).rstrip()
            for row in cells]


def convert(md):
    out, table = [], []
    for line in md.splitlines():
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        if table:
            out += render_table(table)
            table = []
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            title = strip_inline(m.group(2))
            out += ["", title, ("=" if len(m.group(1)) == 1 else "-") * len(title)]
            continue
        out.append(strip_inline(line))
    if table:
        out += render_table(table)
    text = "\n".join(out).strip() + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)              # collapse runs of blanks


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        text = convert(f.read())
    with open(dst, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(text)
