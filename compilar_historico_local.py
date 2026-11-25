"""Ejecuta las celdas y expresiones del QMD en un proceso Python local.

Alternativa de compilación sin kernel TCP: produce Markdown para Quarto/Pandoc.
No sustituye cálculos por números manuales. El QMD también admite quarto render.
"""
import os, sys
if not sys.flags.utf8_mode:
    os.execv(sys.executable, [sys.executable, '-X', 'utf8', *sys.argv])
from pathlib import Path
import ast, re, os, sys, json, hashlib, contextlib, io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Markdown
import cloudpickle

source=Path(sys.argv[1]).resolve()
os.chdir(source.parent)
text=source.read_text(encoding='utf-8-sig')
parts=re.split(r'(```\{python\}\n.*?\n```)',text,flags=re.S)
sys.path.insert(0,str(source.parent))
ns={'__name__':'__article__'}
outputs=[]
cache=Path('.calculos_local.pkl')
use_cache='--reuse' in sys.argv
calc_code=''.join(p for p in parts if p.startswith('```{python}') and '#| include: false' in p)
digest=hashlib.sha256(calc_code.encode()).hexdigest()
if use_cache and cache.exists():
    saved=cloudpickle.loads(cache.read_bytes())
    assert saved['digest']==digest,'El código de cálculo cambió: reejecutar sin --reuse'
    ns=saved['namespace']
figroot=Path(source.stem+'_files/figure-local')
figroot.mkdir(parents=True,exist_ok=True)
def inline(s):
    return re.sub(r'`\{python\}\s+([^`]+)`',lambda m:str(eval(m.group(1),ns)),s)
for part in parts:
    if not part.startswith('```{python}'):
        outputs.append(inline(part))
        continue
    label=re.search(r'#\| label: (.+)',part).group(1)
    hidden='#| include: false' in part
    if hidden and use_cache: continue
    print('Ejecutando',label,flush=True)
    lines=part.splitlines()[1:-1]
    code='\n'.join(x for x in lines if not x.startswith('#|'))
    captured=[]
    def capture(obj):
        if hasattr(obj,'data'): captured.append(str(obj.data))
    def show(*args,**kwargs):
        nums=plt.get_fignums()
        for i,num in enumerate(nums):
            fig=plt.figure(num)
            target=figroot/(label+('' if i==0 else f'-{i}')+'.png')
            fig.savefig(target,bbox_inches='tight')
            cap=re.search(r'#\| fig-cap: "(.*)"',part)
            caption=inline(cap.group(1)) if cap else ''
            captured.append(f'![{caption}]({target.as_posix()}){{#{label}}}')
        plt.close('all')
    plt.show=show
    tree=ast.parse(code)
    last=None
    if tree.body and isinstance(tree.body[-1],ast.Expr):
        last=tree.body.pop().value
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(tree,str(source),'exec'),ns)
        ns['display']=capture
        value=eval(compile(ast.Expression(last),str(source),'eval'),ns) if last is not None else None
    if not hidden:
        if isinstance(value,Markdown): captured.append(value.data)
        rendered='\n\n'.join(captured)
        cap=re.search(r'#\| tbl-cap: "(.*)"',part)
        if cap: rendered+='\n\n: '+inline(cap.group(1))+' {#'+label+'}'
        outputs.append(rendered+'\n')
    elif False:
        clean={k:v for k,v in ns.items() if k!='__builtins__'}
        cache.write_bytes(cloudpickle.dumps({'digest':digest,'namespace':clean}))
rendered=''.join(outputs)
assert not re.search(r'`\{python\}',rendered)
Path(source.stem+'_actualizado.md').write_text(rendered,encoding='utf-8')
print('Markdown generado desde el QMD:',source.stem+'_actualizado.md',flush=True)
