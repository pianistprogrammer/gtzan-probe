"""Create a source/evidence ZIP with no Git history, audio, manuscript, or pickles.

Run from repository root: python scripts/export_public.py
This does not publish, commit, or rewrite the existing remote repository.
"""
from pathlib import Path
import hashlib
import json
import zipfile

root=Path(__file__).resolve().parents[1]
paths=[root/p for p in ['README.md','pyproject.toml','uv.lock','.python-version','.gitignore','LICENSE','regen_all_figures.py','regen_shap_fig.py','regen_xai_summary_table.py']]
for directory in ['src','scripts','tests','docs','notebooks','results']:
    paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and not p.name.startswith('._')
                 and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json','.csv','.ipynb','.yml','.yaml'})
paths=sorted(set(p for p in paths if p.exists()))
release=root/'release';release.mkdir(exist_ok=True)
output=release/'gtzan-probe-public.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
    hashes={}
    for p in paths:
        name=str(p.relative_to(root));data=p.read_bytes()
        z.writestr('gtzan-probe/'+name,data);hashes[name]=hashlib.sha256(data).hexdigest()
    z.writestr('gtzan-probe/EXPORT_SHA256.json',json.dumps(hashes,indent=2)+'\n')
print(f'{output} ({len(paths)} files; no history)')
