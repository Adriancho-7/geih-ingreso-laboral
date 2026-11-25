"""Clasificación del ingreso laboral en cuatro tramos (C1–C4) con CV anidada agrupada.

Modelo final (el único que se despliega y el que describe la ficha): dominio
nacional con la ciudad como predictor (cinco áreas del estudio + "Resto del país").
Las cinco áreas metropolitanas se conservan como modelos de comparación, con el
mismo protocolo y sin la variable ciudad (constante dentro de cada área).

- Cortes P10, P50 y P90 por ciudad y año, calculados solo con el conjunto de ajuste.
- Partición 60/20/20 por hogar-mes; la nacional respeta la de las cinco áreas, de
  modo que la comparación ciudad vs. modelo final usa las mismas personas de prueba.
- Regla de decisión: mínimo costo esperado con la matriz de decision_costos.py.
- Selección: familias probabilísticas por costo esperado; SVM por F1 macro (sus
  márgenes no son probabilidades y no pueden desplegarse con la regla de costos).
"""
from pathlib import Path
import os, json, hashlib, itertools, copy, warnings, sys, shutil, datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, log_loss, silhouette_score
from threadpoolctl import threadpool_limits
from modelado_historico import metricas, NUM, CAT, SEED, CLASES
import decision_costos as dc

# Núcleos para bosques, vecinos y álgebra lineal; no cambia los resultados con semilla fija.
N_JOBS = max(1, int(os.environ.get('GEIH_NJOBS', '2')))
threadpool_limits(N_JOBS)
CAT_LOCAL = [c for c in CAT if c != 'ciudad']
FEATURES = NUM + CAT_LOCAL                 # modelos de comparación por área
FEATURES_FINAL = FEATURES + ['ciudad']     # modelo final nacional
AREAS = ['Medellín AM', 'Barranquilla AM', 'Bogotá', 'Bucaramanga AM', 'Cali AM', 'Nacional']
FINAL = 'Nacional'
CIUDADES = AREAS[:5]
RESTO = 'Resto del país'
ETIQUETA = {**{a: a for a in CIUDADES}, FINAL: 'Nacional + ciudad (final)'}
GRIDS = {'Logistica': [{'C': .1}, {'C': 1.}], 'Arbol': [{'max_depth': 6}, {'max_depth': 12}],
         'Random Forest': [{'max_features': 'sqrt'}, {'max_features': .5}], 'SVM lineal': [{'C': .1}, {'C': 1.}],
         'ACP + SVM': [{'varianza': .8}, {'varianza': .95}], 'KNN': [{'k': k} for k in [5, 15, 31, 51, 75]],
         'Red MLP': [{'alpha': .001, 'capas': [32]}, {'alpha': .01, 'capas': [32, 16]}]}
KNN_CACHE = {}
NEIGHBOR_CACHE = {}


def jsave(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def features_de(area):
    return FEATURES_FINAL if area == FINAL else FEATURES


def preprocessing(feats=FEATURES):
    cats = [c for c in feats if c not in NUM]
    return ColumnTransformer([
        ('num', Pipeline([('imputar', SimpleImputer(strategy='median')), ('escalar', StandardScaler())]), NUM),
        ('cat', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                          ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)])


def inputs(d, feats=FEATURES):
    x = d[feats].copy()
    for c in feats:
        if c not in NUM:
            x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
    return x


def targets(train, other):
    """Cortes P10/P50/P90 por ciudad y año con el conjunto de ajuste; se aplican congelados al otro."""
    cuts = train.groupby(['ciudad', 'anio']).ingreso.quantile([.1, .5, .9]).unstack()
    cuts.columns = ['q10', 'q50', 'q90']

    def label(d):
        key = pd.MultiIndex.from_arrays([d.ciudad.to_numpy(dtype=object), d.anio.to_numpy()])
        q = cuts.reindex(key).to_numpy()
        income = d.ingreso.to_numpy()
        assert np.isfinite(q).all()
        return np.sum(income[:, None] > q, axis=1).astype('int8')
    return label(train), label(other), cuts


def cargar(root, area):
    """Datos del dominio con columna `conjunto`.

    Áreas: partición histórica de resultados_historicos/particion.parquet.
    Nacional: ciudad = área del estudio para sus hogares y "Resto del país" para los
    demás; los hogares de las cinco áreas conservan su conjunto y el resto se divide
    60/20/20 por hogar-mes dentro de cada año, con la semilla del proyecto.
    """
    root = Path(root)
    part = pd.read_parquet(root/'resultados_historicos/particion.parquet')[['id_persona', 'id_hogar', 'ciudad', 'conjunto']]
    if area != FINAL:
        d = pd.read_parquet(root/'resultados_historicos/base_individual.parquet')
        d = d[d.ciudad == area].reset_index(drop=True)
        return d.merge(part[['id_persona', 'conjunto']], on='id_persona', validate='one_to_one')
    d = pd.read_parquet(root/'datos_ml_nacional/base_individual.parquet')
    assert part.groupby('id_hogar')[['ciudad', 'conjunto']].nunique().max().max() == 1
    hog = part.drop_duplicates('id_hogar').set_index('id_hogar')
    d['ciudad'] = d.id_hogar.map(hog.ciudad).fillna(RESTO).astype(str)
    d['conjunto'] = d.id_hogar.map(hog.conjunto)
    # La base de las cinco áreas está contenida de forma idéntica en la nacional.
    assert int(d.conjunto.notna().sum()) == len(part) and d.id_persona[d.conjunto.notna()].isin(part.id_persona).all()
    rng = np.random.default_rng(SEED)
    resto = d.conjunto.isna().to_numpy()
    asignacion = {}
    for _, g in d.loc[resto].groupby('anio', sort=True):
        hogares = np.array(sorted(g.id_hogar.unique()))
        rng.shuffle(hogares)
        n1, n2 = int(.6*len(hogares)), int(.8*len(hogares))
        for etiqueta, ids in [('entrenamiento', hogares[:n1]), ('validacion', hogares[n1:n2]), ('prueba', hogares[n2:])]:
            asignacion.update(dict.fromkeys(ids, etiqueta))
    d.loc[resto, 'conjunto'] = d.loc[resto, 'id_hogar'].map(asignacion)
    d['conjunto'] = d.conjunto.astype(str)
    assert d.conjunto.isin(['entrenamiento', 'validacion', 'prueba']).all()
    assert d.groupby('id_hogar').conjunto.nunique().max() == 1
    return d


def epoca_restaurada(curva):
    mejor = float('inf'); elegida = None
    for fila in curva:
        if fila['perdida_parada'] < mejor-1e-4:
            mejor = fila['perdida_parada']; elegida = int(fila['epoca'])
    return elegida


def fit(name, param, X, y, groups):
    extra = {}; z = X
    if name == 'Logistica': model = LogisticRegression(C=param['C'], max_iter=1500, solver='lbfgs')
    elif name == 'Arbol': model = DecisionTreeClassifier(**param, min_samples_leaf=30, random_state=SEED)
    elif name == 'Random Forest':
        model = RandomForestClassifier(**param, n_estimators=100, max_depth=14, min_samples_leaf=30, n_jobs=N_JOBS, random_state=SEED)
    elif name == 'SVM lineal': model = LinearSVC(**param, dual=False, max_iter=5000, random_state=SEED)
    elif name == 'ACP + SVM':
        pca = PCA(n_components=param['varianza'], svd_solver='full'); z = pca.fit_transform(X); extra['pca'] = pca
        model = LinearSVC(C=1, dual=False, max_iter=5000, random_state=SEED)
    elif name == 'KNN':
        ix = np.arange(len(y))
        if len(ix) > 50000: ix, _ = train_test_split(ix, train_size=50000, stratify=y, random_state=SEED)
        key = (id(X), id(y))
        if key not in KNN_CACHE:
            KNN_CACHE.clear(); NEIGHBOR_CACHE.clear()
            KNN_CACHE[key] = NearestNeighbors(n_neighbors=75, algorithm='kd_tree', leaf_size=60, n_jobs=N_JOBS).fit(X[ix])
        model = KNN_CACHE[key]; extra.update(y=y[ix], n_ajuste=len(ix))
        return dict(model=model, name=name, param=param, **extra)
    elif name == 'Red MLP':
        group_codes = pd.factorize(groups, sort=True)[0]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=SEED).split(X, y, group_codes))
        model = MLPClassifier(hidden_layer_sizes=tuple(param['capas']), activation='relu', solver='adam', alpha=param['alpha'],
                              batch_size=1024, learning_rate_init=.001, max_iter=1, early_stopping=False, random_state=SEED)
        best = None; bestloss = np.inf; patience = 0; curve = []
        for epoch in range(1, 61):
            model.partial_fit(X[a], y[a], classes=np.arange(4)); loss = log_loss(y[b], model.predict_proba(X[b]), labels=np.arange(4))
            curve.append(dict(epoca=epoch, perdida_entreno=model.loss_, perdida_parada=loss))
            if loss < bestloss-1e-4: bestloss = loss; best = copy.deepcopy(model); patience = 0
            else: patience += 1
            if patience >= 6: break
        model = best; extra.update(curva=curve, n_ajuste=len(a), n_parada=len(b), epoca_elegida=epoca_restaurada(curve))
        return dict(model=model, name=name, param=param, **extra)
    else: raise ValueError(name)
    model.fit(z, y)
    return dict(model=model, name=name, param=param, n_ajuste=len(y), **extra)


def scores(bundle, X):
    if 'pca' in bundle: X = bundle['pca'].transform(X)
    model = bundle['model']
    if bundle['name'] == 'KNN':
        key = (id(model), id(X))
        if key not in NEIGHBOR_CACHE:
            NEIGHBOR_CACHE.clear()
            NEIGHBOR_CACHE[key] = np.vstack([model.kneighbors(X[i:i+2500], return_distance=False) for i in range(0, len(X), 2500)])
        v = NEIGHBOR_CACHE[key][:, :bundle['param']['k']]
        return np.stack([(bundle['y'][v] == c).mean(axis=1) for c in range(4)], axis=1)
    return model.predict_proba(X) if hasattr(model, 'predict_proba') else model.decision_function(X)


def criterio(name):
    """Columna de selección y si se minimiza."""
    return ('costo_regla', True) if name in dc.PROBABILISTICOS else ('f1_macro', False)


def evaluar(y, s, name):
    """Métricas de la clase de máxima puntuación y, si hay probabilidades, de la regla de costos."""
    pred = s.argmax(axis=1); r = metricas(y, pred, s); r['costo_argmax'] = dc.costo_medio(y, pred)
    if name in dc.PROBABILISTICOS:
        dec = dc.decidir(s); r.update({k+'_regla': v for k, v in metricas(y, dec).items()}); r['costo_regla'] = dc.costo_medio(y, dec)
    else:
        r['costo_regla'] = np.nan
    return r


def run_domain(root, area):
    root = Path(root); out = root/'resultados_ml_final'/str(AREAS.index(area)); out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(dc.__file__), root/'resultados_historicos/base_individual.parquet', root/'resultados_historicos/particion.parquet']
    if area == FINAL: sources.append(root/'datos_ml_nacional/base_individual.parquet')
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    if (out/'resumen.json').exists() and json.loads((out/'resumen.json').read_text(encoding='utf-8')).get('digest') == digest:
        if area == FINAL and not (root/'modelo_final/configuracion.json').exists(): exportar_modelo_final(root)
        return
    feats = features_de(area); print('ML dominio', area, '| predictores:', ', '.join(feats), '| núcleos:', N_JOBS, flush=True)
    d = cargar(root, area).reset_index(drop=True)
    d[['id_persona', 'id_hogar', 'anio', 'mes', 'ciudad', 'conjunto']].to_parquet(out/'particion.parquet', index=False)
    tr = d[d.conjunto == 'entrenamiento'].reset_index(drop=True); va = d[d.conjunto == 'validacion'].reset_index(drop=True); te = d[d.conjunto == 'prueba'].reset_index(drop=True)
    assert set(tr.id_hogar).isdisjoint(va.id_hogar) and set(tr.id_hogar).isdisjoint(te.id_hogar) and set(va.id_hogar).isdisjoint(te.id_hogar)
    folds = list(GroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(tr, groups=pd.factorize(tr.id_hogar, sort=True)[0]))
    grid = []; outer = []; assignment = []; alerts = []
    for fold, (a, b) in enumerate(folds, 1):
        A = tr.iloc[a]; B = tr.iloc[b]; assignment.extend(dict(id_persona=v, pliegue=fold) for v in B.id_persona)
        inner = list(GroupKFold(n_splits=2, shuffle=True, random_state=SEED+fold).split(A, groups=pd.factorize(A.id_hogar, sort=True)[0]))
        local = []
        for innerfold, (u, v) in enumerate(inner, 1):
            KNN_CACHE.clear(); NEIGHBOR_CACHE.clear()
            U = A.iloc[u]; V = A.iloc[v]; yt, yv, _ = targets(U, V); pre = preprocessing(feats)
            xt = pre.fit_transform(inputs(U, feats)).astype('float32'); xv = pre.transform(inputs(V, feats)).astype('float32')
            for name, params in GRIDS.items():
                print(area, fold, innerfold, name, flush=True)
                for param in params:
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always'); model = fit(name, param, xt, yt, U.id_hogar); score = scores(model, xv)
                    alerts.extend(dict(modelo=name, etapa=f'CV{fold}.{innerfold}', aviso=str(w.message)) for w in ws)
                    row = dict(modelo=name, pliegue=fold, interno=innerfold, param=json.dumps(param, sort_keys=True), **evaluar(yv, score, name))
                    grid.append(row); local.append(row)
            print(area, 'CV interna', fold, innerfold, flush=True)
            pd.DataFrame(grid).to_csv(out/'cv_busqueda.csv', index=False)
        KNN_CACHE.clear(); NEIGHBOR_CACHE.clear()
        yt, yb, _ = targets(A, B); pre = preprocessing(feats)
        xt = pre.fit_transform(inputs(A, feats)).astype('float32'); xb = pre.transform(inputs(B, feats)).astype('float32')
        localdf = pd.DataFrame(local)
        for name in GRIDS:
            col, asc = criterio(name)
            p = (localdf[localdf.modelo == name].groupby('param')[col].mean().reset_index()
                 .sort_values([col, 'param'], ascending=[asc, True]).iloc[0]['param'])
            model = fit(name, json.loads(p), xt, yt, A.id_hogar); s = scores(model, xb)
            outer.append(dict(modelo=name, pliegue=fold, param=p, **evaluar(yb, s, name)))
        pd.DataFrame(outer).to_csv(out/'cv_externa.csv', index=False)
    pd.DataFrame(assignment).to_parquet(out/'pliegues.parquet', index=False)
    # Configuración final: moda de las elegidas en los pliegues externos; desempate por el criterio interno medio.
    configs = {}; odf = pd.DataFrame(outer); gdf = pd.DataFrame(grid)
    for name in GRIDS:
        col, asc = criterio(name)
        votes = odf[odf.modelo == name].param.value_counts(); options = votes[votes == votes.max()].index
        best = gdf[(gdf.modelo == name) & gdf.param.isin(options)].groupby('param')[col].mean().sort_values(ascending=asc, kind='stable').index[0]
        configs[name] = json.loads(best)
    KNN_CACHE.clear(); NEIGHBOR_CACHE.clear()
    yt, yv, cuts = targets(tr, va); _, ye, _ = targets(tr, te); cuts.to_csv(out/'cortes.csv')
    pre = preprocessing(feats); xt = pre.fit_transform(inputs(tr, feats)).astype('float32'); xv = pre.transform(inputs(va, feats)).astype('float32')
    joblib.dump(pre, out/'preprocesador.joblib'); validation = []; bundles = {}; valscores = {}
    for i, name in enumerate(GRIDS):
        model = fit(name, configs[name], xt, yt, tr.id_hogar); s = scores(model, xv); validation.append(dict(modelo=name, **evaluar(yv, s, name)))
        if name in dc.PROBABILISTICOS: valscores[name] = s.astype('float32')
        joblib.dump(model, out/f'modelo_{i}.joblib'); bundles[name] = i
        jsave(out/f'parametros_{i}.json', dict(modelo=name, elegidos=configs[name], estimator=model['model'].get_params(), n_ajuste=model.get('n_ajuste'),
              epoca=model.get('epoca_elegida'), componentes=model.get('pca').n_components_ if 'pca' in model else None))
        if name == 'Red MLP': pd.DataFrame(model['curva']).to_csv(out/'red_aprendizaje.csv', index=False)
    val = pd.DataFrame(validation); val.to_csv(out/'validacion.csv', index=False)
    winner = val[val.modelo.isin(dc.PROBABILISTICOS)].sort_values(['costo_regla', 'f1_macro', 'modelo'], ascending=[True, False, True]).modelo.iloc[0]
    # Sensibilidad del costo de excluir a C1 (solo validación; la matriz no se elige con datos).
    sens = []
    for prioridad in dc.PRIORIDADES:
        C = dc.matriz(prioridad); dec = dc.decidir(valscores[winner], C); t = dc.tasas_ovr(yv, dec)
        sens.append(dict(prioridad=prioridad, tau=1/(1+prioridad), costo_propio=dc.costo_medio(yv, dec, C), costo_principal=dc.costo_medio(yv, dec),
                         **metricas(yv, dec), **{f'asignados_C{c+1}': float((dec == c).mean()) for c in range(4)},
                         **{f'TPR_C{c+1}': t.TPR[c] for c in range(4)}, **{f'VPP_C{c+1}': t.VPP[c] for c in range(4)}))
    pd.DataFrame(sens).to_csv(out/'sensibilidad_costos.csv', index=False); del valscores
    jsave(out/'decision_pre_prueba.json', dict(ganador=winner, configuraciones=configs, predictores=feats,
          matriz_costos=dc.COSTOS.tolist(), prioridad_C1=dc.PRIORIDAD, tau_C1=dc.TAU,
          criterio='Familia probabilística de menor costo esperado en la validación externa (20 %) con la regla de costos; '
                   'desempate por F1 macro. Hiperparámetros por CV anidada 3x2 del 60 %: costo esperado (probabilísticas) o F1 macro (SVM).'))
    xe = pre.transform(inputs(te, feats)).astype('float32'); metricsrows = []; cms = []; curves = []; examples = []; profiles = []; subgroups = []; preds = []
    for name, i in bundles.items():
        model = joblib.load(out/f'modelo_{i}.joblib'); s = scores(model, xe); pr = s.argmax(axis=1)
        metricsrows.append(dict(modelo=name, ganador=name == winner, probabilistico=name in dc.PROBABILISTICOS, **evaluar(ye, s, name)))
        dec = dc.decidir(s) if name in dc.PROBABILISTICOS else None
        for regla, z in [('argmax', pr), ('costos', dec)]:
            if z is None: continue
            cm = confusion_matrix(ye, z, labels=np.arange(4))
            cms.extend(dict(modelo=name, regla=regla, real=a, predicha=b, n=int(cm[a, b])) for a, b in itertools.product(range(4), repeat=2))
        # Las curvas se reducen solo para dibujar; AUC y métricas usan toda la prueba.
        for c in range(4):
            fpr, tpr, _ = roc_curve(ye == c, s[:, c]); precision, recall, _ = precision_recall_curve(ye == c, s[:, c])
            for kind, x, y in [('ROC', fpr, tpr), ('PR', recall, precision)]:
                ix = np.unique(np.linspace(0, len(x)-1, min(250, len(x))).astype(int)); curves.extend(dict(modelo=name, clase=c, tipo=kind, x=x[j], y=y[j]) for j in ix)
        preds.append(pd.DataFrame(dict(id_persona=te.id_persona, real=ye, predicha=pr, predicha_regla=dec if dec is not None else np.full(len(ye), -1, dtype='int8'), modelo=name)))
        if name == winner:
            prob = pd.DataFrame(s.astype('float32'), columns=[f'p_C{c+1}' for c in range(4)]); prob.insert(0, 'id_persona', te.id_persona.to_numpy())
            prob.to_parquet(out/'probabilidades_ganador.parquet', index=False); dc.calibracion(ye, s).to_csv(out/'calibracion.csv', index=False)
            result = te.copy(); result['real'] = ye; result['predicha'] = dec; result['predicha_argmax'] = pr
            result['grupo_edad'] = pd.cut(result.edad, [17, 29, 44, 59, 200], labels=dc.EDADES).astype(object)
            for c, g in result.groupby('predicha'):
                profiles.append(dict(clase=int(c), n=len(g), edad_mediana=g.edad.median(), ingreso_mediano=g.ingreso.median(), educacion_modal=g.educacion.mode().iloc[0]))
                for j, (_, v) in enumerate(g.head(2).iterrows()):
                    examples.append(dict(caso=f'Caso-{c+1}-{j+1}', ciudad=v.ciudad, anio=v.anio, edad=v.edad, educacion=v.educacion, horas=v.horas, antiguedad=v.antiguedad,
                                         real=int(v.real)+1, predicha=int(c)+1, max_probabilidad=int(v.predicha_argmax)+1))
            variables = ['sexo', 'grupo_edad', 'educacion'] + (['ciudad'] if area == FINAL else [])
            for var in variables:
                for value, g in result.groupby(var): subgroups.append(dict(variable=var, grupo=value, n=len(g), **metricas(g.real, g.predicha), costo=dc.costo_medio(g.real, g.predicha)))
            tg = [dc.tasas_por_grupo(result, var, 'real', col).assign(regla=regla) for regla, col in [('costos', 'predicha'), ('argmax', 'predicha_argmax')] for var in variables]
            tg = pd.concat(tg, ignore_index=True); tg.to_csv(out/'tasas_grupo.csv', index=False)
            pd.concat([dc.brechas(tg[tg.regla == r]).assign(regla=r) for r in ['costos', 'argmax']]).to_csv(out/'brechas_grupo.csv', index=False)
            pd.concat([dc.tasas_ovr(ye, dec).assign(regla='costos'), dc.tasas_ovr(ye, pr).assign(regla='argmax')]).to_csv(out/'tasas_clase.csv', index=False)
    freq = np.bincount(yt, minlength=4)/len(yt); mayor = int(freq.argmax()); constante = int((freq @ dc.COSTOS).argmin())
    base = np.repeat(mayor, len(ye)); metricsrows.append(dict(modelo='Mayoritaria', ganador=False, probabilistico=False, **metricas(ye, base), costo_argmax=dc.costo_medio(ye, base), costo_regla=np.nan))
    base = np.repeat(constante, len(ye))
    metricsrows.append(dict(modelo='Constante de costo', ganador=False, probabilistico=False, **{k+'_regla': v for k, v in metricas(ye, base).items()}, costo_argmax=np.nan, costo_regla=dc.costo_medio(ye, base)))
    for fname, rows in [('prueba', metricsrows), ('confusiones', cms), ('curvas', curves), ('ejemplos', examples), ('perfiles', profiles), ('subgrupos', subgroups), ('avisos', alerts)]:
        pd.DataFrame(rows).to_csv(out/(fname+'.csv'), index=False)
    pd.concat(preds).to_parquet(out/'predicciones.parquet', index=False)
    # ACP y clústeres: ajuste sin ingreso ni clase, selección de k por silhouette de validación.
    pca = PCA(svd_solver='full').fit(xt)
    pd.DataFrame(dict(componente=np.arange(1, len(pca.explained_variance_ratio_)+1), varianza=pca.explained_variance_ratio_)).to_csv(out/'acp.csv', index=False)
    rng = np.random.default_rng(SEED); sample = np.sort(rng.choice(len(xv), min(2000, len(xv)), replace=False)); krows = []; bestk = None; bestscore = -1
    for k in [2, 3, 4, 5, 6]:
        model = KMeans(n_clusters=k, n_init=3, max_iter=150, random_state=SEED).fit(xt); sil = silhouette_score(xv[sample], model.predict(xv[sample])); krows.append(dict(k=k, silhouette=sil, inercia=model.inertia_))
        if sil > bestscore: bestscore = sil; bestk = k; joblib.dump(model, out/'kmeans.joblib')
    pd.DataFrame(krows).to_csv(out/'clusters.csv', index=False)
    cluster = joblib.load(out/'kmeans.joblib').predict(xe); pt = pca.transform(xe); sample = np.sort(rng.choice(len(xe), min(5000, len(xe)), replace=False))
    pd.DataFrame(dict(cp1=pt[sample, 0], cp2=pt[sample, 1], cluster=cluster[sample])).to_csv(out/'clusters_grafico.csv', index=False)
    jsave(out/'resumen.json', dict(digest=digest, area=area, etiqueta=ETIQUETA[area], n=len(d), entrenamiento=len(tr), validacion=len(va), prueba=len(te),
          pliegues_externos=3, internos=2, ganador=winner, k=bestk, features=feats, matriz_costos=dc.COSTOS.tolist(), tau=dc.TAU, nucleos=N_JOBS))
    print('ML TERMINADO', area, winner, flush=True)
    if area == FINAL: exportar_modelo_final(root)


def ficha_filas(root):
    """Ficha del modelo final (campo, contenido) a partir de los resultados guardados."""
    root = Path(root); p = root/'resultados_ml_final'/str(AREAS.index(FINAL))
    s = json.loads((p/'resumen.json').read_text(encoding='utf-8')); dec = json.loads((p/'decision_pre_prueba.json').read_text(encoding='utf-8'))
    q = pd.read_csv(p/'prueba.csv').set_index('modelo'); w = dec['ganador']; r = q.loc[w]; cte = q.loc['Constante de costo']
    t = pd.read_csv(p/'tasas_clase.csv'); t = t[t.regla == 'costos'].set_index('clase')
    par = json.dumps(dec['configuraciones'][w], ensure_ascii=False)
    miles = lambda v: f'{int(v):,}'.replace(',', ' ')
    tasas = '; '.join(f"{c}: TPR {t.loc[c, 'TPR']:.2f}, FPR {t.loc[c, 'FPR']:.2f}, VPP {t.loc[c, 'VPP']:.2f}" for c in dc.NOMBRES)
    return [
        ['Nombre y versión', 'Clasificador GEIH de tramos de ingreso laboral · modelo final · versión 2026-11-25'],
        ['Modelo desplegado', f'{w} {par}, entrenado con el 60 % nacional (n = {miles(s["entrenamiento"])}); es el único que se despliega (carpeta modelo_final/ y cargar_y_predecir.py)'],
        ['Pregunta', 'Asignar a un asalariado uno de cuatro tramos relativos de ingreso laboral: C1 ≤ P10; P10 < C2 ≤ P50; P50 < C3 ≤ P90; C4 > P90'],
        ['Cortes', 'P10, P50 y P90 por ciudad y año, calculados solo con entrenamiento (archivo modelo_final/cortes_entrenamiento.csv)'],
        ['Datos', f'GEIH 2008–2025, base nacional armonizada, sin ponderar: {miles(s["n"])} personas-mes; 60/20/20 por hogar-mes'],
        ['Predictores', 'edad, antigüedad, horas, sexo, jefatura, estudia, estado civil, educación y ciudad (cinco áreas + Resto del país); el ingreso nunca es predictor'],
        ['Uso previsto', 'Priorizar el contenido de una campaña informativa de orientación laboral y financiera cuando el ingreso no se declara; siempre revisable por la persona'],
        ['Usos no previstos', 'Negar o asignar empleo, crédito, subsidios o beneficios; estimar el ingreso en COP; decisiones automáticas sin revisión humana'],
        ['Regla de decisión', f'Mínimo costo esperado con la matriz de costos: asignar un tramo superior a una persona de C1 cuesta {dc.PRIORIDAD:g} por tramo; los demás errores cuestan la distancia entre tramos. Se asigna C1 si y solo si P(C1) ≥ {dc.TAU:.2f}, el umbral de la versión binaria'],
        ['Selección', 'Familia probabilística de menor costo esperado en validación externa; hiperparámetros por CV anidada 3x2; prueba usada una sola vez'],
        ['Desempeño en prueba', f'Costo {1000*r.costo_regla:.0f} por mil personas (constante: {1000*cte.costo_regla:.0f}); F1 macro {r.f1_macro_regla:.3f}; exactitud {r.exactitud_regla:.3f}; AUC macro {r.auc_macro:.3f}'],
        ['Tasas por clase (prueba)', tasas],
        ['Equidad', 'TPR, FPR y VPP por clase frente a las demás según sexo, edad, educación y ciudad; las brechas se informan, no se corrigen'],
        ['Limitaciones', 'Muestra no ponderada de asalariados con contrato escrito; asociaciones, no causas; cortes de años pasados; sexo es predictor; calibración no garantizada fuera de la GEIH'],
        ['Responsables', 'Autores del proyecto; revisión humana obligatoria antes de cualquier uso'],
    ]


def exportar_modelo_final(root):
    """Empaqueta el modelo final que se despliega: preprocesador, modelo, cortes, matriz y ficha."""
    root = Path(root); src = root/'resultados_ml_final'/str(AREAS.index(FINAL)); dst = root/'modelo_final'; dst.mkdir(exist_ok=True)
    dec = json.loads((src/'decision_pre_prueba.json').read_text(encoding='utf-8')); name = dec['ganador']; i = list(GRIDS).index(name)
    joblib.dump(joblib.load(src/'preprocesador.joblib'), dst/'preprocesador.joblib', compress=3)
    joblib.dump(joblib.load(src/f'modelo_{i}.joblib'), dst/'modelo.joblib', compress=3)
    shutil.copyfile(src/'cortes.csv', dst/'cortes_entrenamiento.csv')
    q = pd.read_csv(src/'prueba.csv').set_index('modelo').loc[name]
    jsave(dst/'configuracion.json', dict(
        nombre='Clasificador GEIH de tramos de ingreso laboral', version='2026-11-25', dominio='Nacional con la ciudad como predictor',
        familia=name, parametros=dec['configuraciones'][name], predictores=FEATURES_FINAL, categorias_ciudad=CIUDADES+[RESTO], clases=CLASES,
        matriz_costos=dc.COSTOS.tolist(), prioridad_C1=dc.PRIORIDAD, tau_C1=dc.TAU,
        regla='Clase de mínimo costo esperado: argmin_k sum_j p_j C[j,k]; empates hacia el tramo menor', semilla=SEED,
        prueba=dict(costo_por_mil=1000*q.costo_regla, f1_macro=q.f1_macro_regla, exactitud=q.exactitud_regla, auc_macro=q.auc_macro),
        sklearn=__import__('sklearn').__version__, generado=datetime.date.today().isoformat()))
    rows = ficha_filas(root)
    text = '# Ficha del modelo final\n\n| Campo | Contenido |\n|:--|:--|\n' + '\n'.join(f'| {a} | {b} |' for a, b in rows) + '\n'
    text += '\n## Matriz de costos (fila = clase real, columna = clase asignada)\n\n| Real \\ Asignada | C1 | C2 | C3 | C4 |\n|:--|--:|--:|--:|--:|\n'
    text += '\n'.join(f'| C{j+1} | ' + ' | '.join(f'{v:g}' for v in dc.COSTOS[j]) + ' |' for j in range(4)) + '\n'
    (dst/'FICHA_MODELO.md').write_text(text, encoding='utf-8')
    print('Modelo final exportado en', dst, flush=True)


if __name__ == '__main__':
    root = sys.argv[1]
    for area in (AREAS if len(sys.argv) < 3 else [sys.argv[2]]): run_domain(root, area)
