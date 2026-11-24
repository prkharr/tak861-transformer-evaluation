"""Read-only layout, navigation, immutable notebook and embedded-source checks."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def literal_assignments(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            try:
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict' and not node.value.args:
                    value = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
                else:
                    value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def validate():
    expected = {
        'established_experiments': [
            '01_tensor_initialization_AutoML.ipynb', '02_patient_level_split_AutoML.ipynb',
            '03_transformer_training_AutoML.ipynb', '04_transformer_evaluation_AutoML.ipynb',
            '05_model_comparison_AutoML.ipynb', '06_time_aware_lightgbm_AutoML.ipynb'],
        'final_challenger': ['01_input_audit.ipynb', '02_tuning_and_blend.ipynb', '03_validation_report.ipynb'],
        'utilities': ['01_gather_and_select_features.ipynb', '02_experiment_tracker.ipynb'],
    }
    paths = {f'notebooks/{folder}/{name}' for folder, names in expected.items() for name in names}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT/'notebooks').rglob('*.ipynb')}
    assert actual == paths, (actual - paths, paths - actual)
    index = json.loads((ROOT/'archive/path_index.json').read_text())
    immutable = [r for r in index['moves'] if Path(r['new']).suffix in ('.ipynb', '.pptx')]
    for row in immutable:
        raw = (ROOT/row['new']).read_bytes()
        unchanged = hashlib.sha256(raw).hexdigest() == row['sha256_before']
        # Windows Git checkouts can change physical line endings, not cell contents.
        if not unchanged and row['new'].endswith('.ipynb'):
            unchanged = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == row['git_blob_sha256']
        assert unchanged, row['new']
    cells = 0
    for name in sorted(paths):
        book = json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
        assert book['nbformat'] == 4
        values = {}
        for cell in book['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            values.update(literal_assignments(source))
            assert not cell.get('outputs') and cell.get('execution_count') is None, name
            cells += 1
        # Tracker uses discovery settings rather than the model notebook contract.
        if 'sf_options' in values:
            assert len(values['sf_options']) == 7 and all(v == '' for v in values['sf_options'].values()), name
        if '/final_challenger/' in name:
            assert values['RUN_ID'] == 'R20261030_NEW' and values['RUN_MODE'] == 'SUMMARY', name
            sources = values['CHALLENGER_SOURCES']
            assert hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() == values['CHALLENGER_SHA'], name
            for source in sources.values():
                ast.parse(source)
            assert values['CHALLENGER_SETTINGS']['seconds'] == 28800, name
    # These are maintained entry points, rather than immutable historical receipts.
    docs = [ROOT/'README.md', ROOT/'archive/README.md', ROOT/'archive/path_index.md', ROOT/'reports/experiment_summary.md']
    docs += [ROOT/'notebooks'/folder/'README.md' for folder in expected]
    links = 0
    for p in docs:
        for href in re.findall(r'\]\(([^\s)]+)\)', p.read_text(encoding='utf-8')):
            if re.match(r'(?:[a-z]+:|#)', href, re.I):
                continue
            assert (p.parent/unquote(href.split('#')[0])).exists(), (str(p), href)
            links += 1
    return dict(active_notebooks=len(paths), preserved_notebooks_and_decks=len(immutable),
                code_cells_compiled=cells, navigation_links_checked=links,
                embedded_challenger_checksums='pass', blank_connections='pass',
                notebook_bytes='unchanged', private_data_execution=False)


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
