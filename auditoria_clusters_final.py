"""Estabilidad de inicialización y perfiles originales de K-means por dominio."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from ml_final import AREAS, cargar, features_de, inputs
threadpool_limits(2)


def run(root):
    root = Path(root)
    for i, area in enumerate(AREAS):
        p = root/'resultados_ml_final'/str(i); marker = p/'auditoria_clusters.json'
        signature = hashlib.sha256(Path(__file__).read_bytes()+(p/'resumen.json').read_bytes()+(p/'kmeans.joblib').read_bytes()).hexdigest()
        if marker.exists() and json.loads(marker.read_text(encoding='utf-8'))['digest'] == signature: continue
        feats = features_de(area); d = cargar(root, area)
        pre = joblib.load(p/'preprocesador.joblib'); base = joblib.load(p/'kmeans.joblib')
        tr = d[d.conjunto == 'entrenamiento']; va = d[d.conjunto == 'validacion'].sample(min(5000, int((d.conjunto == 'validacion').sum())), random_state=78018); te = d[d.conjunto == 'prueba'].copy()
        X = pre.transform(inputs(tr, feats)).astype('float32'); V = pre.transform(inputs(va, feats)).astype('float32'); original = base.predict(V); rows = []
        for seed in [78019, 78020]:
            other = KMeans(n_clusters=base.n_clusters, n_init=3, max_iter=150, random_state=seed).fit(X)
            rows.append(dict(semilla=seed, ARI=adjusted_rand_score(original, other.predict(V)), n_entrenamiento=len(X), n_validacion=len(V)))
        pd.DataFrame(rows).to_csv(p/'clusters_estabilidad.csv', index=False)
        te['cluster'] = base.predict(pre.transform(inputs(te, feats)).astype('float32'))+1
        # Clase observada con los cortes de entrenamiento por ciudad y año (descripción posterior).
        cuts = pd.read_csv(p/'cortes.csv').set_index(['ciudad', 'anio'])
        q = cuts.reindex(pd.MultiIndex.from_arrays([te.ciudad.to_numpy(dtype=object), te.anio.to_numpy()])).to_numpy()
        te['clase'] = np.sum(te.ingreso.to_numpy()[:, None] > q, axis=1)+1
        rows = []
        for cluster, g in te.groupby('cluster'):
            rows.append(dict(cluster=int(cluster), n=len(g), edad_mediana=g.edad.median(), antiguedad_mediana=g.antiguedad.median(), horas_medianas=g.horas.median(),
                             ingreso_mediano=g.ingreso.median(), educacion_modal=g.educacion.mode().iloc[0], **{f'prop_C{c}': float(g.clase.eq(c).mean()) for c in range(1, 5)}))
        pd.DataFrame(rows).to_csv(p/'clusters_perfiles.csv', index=False)
        marker.write_text(json.dumps(dict(digest=signature, semillas=[78018, 78019, 78020], tipo='Estabilidad de inicialización, no bootstrap de muestra')), encoding='utf8')
        print('Clúster auditado', area, flush=True)


if __name__ == '__main__': run(sys.argv[1] if len(sys.argv) > 1 else Path.cwd())
