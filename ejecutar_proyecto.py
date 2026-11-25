"""Punto de entrada portable del proyecto; ejecutar desde VS Code o terminal.

Calcula el modelo final (nacional con ciudad) y los cinco modelos de comparación por área,
las regresiones y las series de tiempo, verifica los resultados y genera el PDF de cada
QMD presente (GEIH_MACHINE_LEARNING.qmd y/o GEIH_SERIES_TIEMPO.qmd).
"""
from pathlib import Path
import sys, os, subprocess, shutil, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recalcular', action='store_true', help='Recalcular todos los modelos; puede tardar varias horas')
    ap.add_argument('--solo-informe', action='store_true', help='Usar resultados guardados y regenerar figuras/PDF')
    ap.add_argument('--qmd', nargs='*', help='QMD a compilar (por defecto, todos los GEIH_*.qmd de la carpeta)')
    a = ap.parse_args()
    root = Path(__file__).resolve().parent; os.chdir(root)
    import json, platform
    from importlib.metadata import version
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'arch', 'prophet', 'torch', 'matplotlib', 'pyarrow', 'dane-data']
    (root/'entorno_final.json').write_text(json.dumps({'python': sys.version, 'plataforma': platform.platform(), 'paquetes': {p: version(p) for p in packages}}, indent=2), encoding='utf-8')
    with (root/'pip_freeze_final.txt').open('w', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    if a.recalcular and a.solo_informe: ap.error('Opciones incompatibles')
    if a.recalcular:
        # Solo borrar marcadores de caché identificados dentro de esta carpeta.
        for folder in ['resultados_ml_final', 'resultados_regresion_final', 'resultados_series', 'resultados_series_final']:
            for file in (root/folder).rglob('*.json'):
                if file.name in ['resumen.json', 'completo.json']:
                    assert file.resolve().is_relative_to(root); file.unlink()
    if not a.solo_informe:
        from ml_final import run_domain, AREAS
        for area in AREAS: run_domain(root, area)          # cinco áreas de comparación + modelo final nacional
        from auditoria_clusters_final import run as clusters
        clusters(root)
        from regresion_final import run as regress
        regress(root)
        from series_temporales import run as series
        series(root)
        from series_final import run as intervals
        intervals(root)
        from parametros_series_final import run as parameters
        parameters(root)
    from verificar_final import run as verify
    verify(root)
    env = os.environ.copy(); env['PYTHONUTF8'] = '1'; env['GEIH_SOLO_INFORME'] = '1'; tiny = Path.home()/'AppData/Roaming/TinyTeX/bin/windows'
    if tiny.exists(): env['PATH'] = str(tiny)+os.pathsep+env['PATH']
    quarto = shutil.which('quarto')
    if not quarto:
        candidate = Path.home()/'AppData/Local/Programs/Quarto/bin/quarto.exe'
        if candidate.exists(): quarto = str(candidate)
    if not quarto: raise RuntimeError('Instale Quarto y TinyTeX; consulte LEEME_PROFESOR.md')
    qmds = [root/q for q in a.qmd] if a.qmd else sorted(root.glob('GEIH_*.qmd'))
    if not qmds: raise RuntimeError('No se encontró ningún GEIH_*.qmd en la carpeta del proyecto')
    for qmd in qmds:
        subprocess.run([sys.executable, '-X', 'utf8', str(root/'compilar_historico_local.py'), str(qmd)], check=True, env=env)
        subprocess.run([quarto, 'render', qmd.stem+'_actualizado.md', '--to', 'pdf'], check=True, env=env)
        print('PDF generado:', root/(qmd.stem+'.pdf'), flush=True)
    print('PROYECTO FINAL TERMINADO.', flush=True)


if __name__ == '__main__': main()
