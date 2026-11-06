"""Мини-конвертер: .py с маркерами `# %%` -> .ipynb (формат jupytext-percent).

Так ноутбуки удобно держать в git: в репозитории лежит читаемый .py,
а .ipynb собирается командой  `python src/py2nb.py notebooks/_churn_analysis.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf


def convert(src: Path, dst: Path) -> None:
    cells, buf, kind = [], [], "code"

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in text.splitlines()
            )
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            buf, kind = [], "markdown" if "[markdown]" in line else "code"
            continue
        buf.append(line)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    dst.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"{src.name} -> {dst.name}  ({len(cells)} ячеек)")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        convert(p, p.with_name(p.name.lstrip("_")).with_suffix(".ipynb"))
