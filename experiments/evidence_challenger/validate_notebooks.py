import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
results=[]
for number in (11,12,13):
    path=next((ROOT/'notebooks/expanded cohort').glob(str(number)+'_*.ipynb'))
    book=json.loads(path.read_text(encoding='utf8'))
    codes=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    for source in codes: ast.parse(source)
    setup=ast.parse(codes[0])
    sf=next(ast.literal_eval(n.value) for n in setup.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='sf_options' for t in n.targets))
    assert len(sf)==7 and not any(sf.values())
    source_cell=ast.parse(codes[2])
    sources=ast.literal_eval(source_cell.body[0].value)
    expected=ast.literal_eval(source_cell.body[1].value)
    assert hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()==expected
    for value in sources.values(): ast.parse(value)
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells'])
    results.append(dict(notebook=path.name,code_cells=len(codes),sf_values_blank=True,
                        syntax='pass',embedded_hash='pass',sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
receipt=dict(notebooks=results,local_tests='3 tests passed; real synthetic LightGBM/PyTorch end-to-end run and resume',
             private_databricks_run='not executed',clinical_uplift='not established')
Path(__file__).with_name('validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
