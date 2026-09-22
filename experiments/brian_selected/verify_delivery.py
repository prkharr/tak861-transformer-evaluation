"""Verify generated notebook formatting, cell compilation and packaging."""
from pathlib import Path
import zipfile
import nbformat

repo = Path(__file__).resolve().parents[2]
folder = repo.parent / "outputs" / "Brian_Selected_Features"
notebooks = sorted(folder.glob("*.ipynb"))
assert len(notebooks) == 4
total = 0
for path in notebooks:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) == 8
    for cell in notebook.cells:
        assert cell.cell_type == "code" and cell.execution_count is None and not cell.outputs
        compile(cell.source, str(path), "exec")
        assert not any(s in cell.source for s in ("sfPassword", "sfToken", "BEGIN PRIVATE KEY", "password ="))
    total += len(notebook.cells)
    mirror = repo / "notebooks" / "Brian Selected Features" / path.name
    assert path.read_bytes() == mirror.read_bytes()
    print(path.name, path.stat().st_size, "bytes")
archive = folder.parent / "Brian_Selected_Features_4_Notebooks.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert set(z.namelist()) == {f.name for f in notebooks} | {"START_HERE.md", "VALIDATION.md"}
    for path in folder.iterdir():
        if path.name in z.namelist():
            assert path.read_bytes() == z.read(path.name)
print(f"Validated {len(notebooks)} notebooks, {total} code cells, matching mirrors and ZIP contents.")
