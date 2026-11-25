"""Construcción reproducible de microdatos GEIH y comparación multiclase.
No estima ni pronostica el monto del salario.
"""
from pathlib import Path
import json, hashlib, time, gc, warnings, platform
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import lector_geih as lector

SEED = 78018
NUM = ['edad', 'antiguedad', 'horas']
CAT = ['sexo', 'jefe', 'estudia', 'estado_civil', 'educacion', 'ciudad']
FEATURES = NUM + CAT
CLASES = ['C1: hasta P10','C2: P10-P50','C3: P50-P90','C4: sobre P90']
EXTRAS = {'P6020','P3271','P6050','P6070','P6170','P6210','P3042','P3069'}

def guardar_json(obj, ruta):
    Path(ruta).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

def construir(root):
    root = Path(root)
    out = root/'resultados_historicos'
    out.mkdir(exist_ok=True)
    base = out/'base_individual.parquet'
    # La caché se identifica por código y resumen de fuentes disponibles.
    rawroot = root.parents[1]/'microdatos_GEIH'
    if not rawroot.exists():
        rawroot = Path.home()/'Downloads/microdatos_GEIH'
    firma = hashlib.sha256(Path(__file__).read_bytes()+Path(lector.__file__).read_bytes()
        +(root/'datos_historicos/mensual.csv').read_bytes()).hexdigest()
    fuentes = lector.dd.geih.find_months(range(2007,2026), root=rawroot)
    estado_fuentes = [(str(f),f.stat().st_size,f.stat().st_mtime_ns)
                      for m in fuentes for f in m.carpeta.rglob('*') if f.is_file()]
    firma = hashlib.sha256((firma+json.dumps(estado_fuentes)).encode()).hexdigest()
    meta = out/'base_manifest.json'
    if base.exists() and meta.exists() and json.loads(meta.read_text())['firma'] == firma:
        print('Base individual: reutilizando caché verificada', flush=True)
        return pd.read_parquet(base)
    lector.VARIABLES.update(EXTRAS)
    esperado = pd.read_csv(root/'datos_historicos/mensual.csv')
    esperado = esperado.loc[esperado.ciudad.ne('Nacional') & esperado.n.notna()]
    parts, aud, disponibilidad = [], [], []
    for m in fuentes:
        ref = esperado.loc[esperado.anio.eq(m.anio) & esperado.mes.eq(m.mes)]
        if ref.empty:
            continue
        if m.mes == int(esperado.loc[esperado.anio.eq(m.anio),'mes'].min()):
            print(f'Construyendo microdatos {m.anio}', flush=True)
        archivos, origen = lector.fuentes(m, True)
        raw = lector.leer_union(archivos)
        if 'CLASE' not in raw:
            nacional = lector.leer_union(m.archivos)
            keys = ['DIRECTORIO','SECUENCIA_P','ORDEN']
            if 'HOGAR' in raw and 'HOGAR' in nacional: keys.insert(2,'HOGAR')
            raw = raw.merge(nacional[keys+['CLASE']], on=keys, how='left', validate='one_to_one',indicator=True)
            assert raw['_merge'].eq('both').all() and raw.CLASE.notna().all()
            raw = raw.drop(columns='_merge')
            del nacional
        n_leidos = len(raw)
        d = lector.seleccionar(raw)
        d = d.loc[d.AREA.isin(lector.AREAS) & d.CLASE.eq(1)].copy()
        assert d.groupby('AREA').size().rename(index=lector.AREAS).to_dict() == ref.set_index('ciudad').n.astype(int).to_dict(), (m.anio,m.mes)
        def campo(nombre):
            return pd.to_numeric(d[nombre], errors='coerce') if nombre in d else pd.Series(np.nan,index=d.index)
        for variable in EXTRAS:
            disponibilidad.append(dict(anio=m.anio,mes=m.mes,variable=variable,
                existe=variable in d,validos=int(campo(variable).notna().sum()),n=len(d)))
        nuevo = campo('P3042')
        viejo = campo('P6210')
        edu_n = nuevo.map({1:'Primaria o menos',2:'Primaria o menos',3:'Primaria o menos',4:'Secundaria',
            5:'Media',6:'Media',7:'Media',**{k:'Superior' for k in range(8,14)}})
        edu_v = viejo.map({1:'Primaria o menos',2:'Primaria o menos',3:'Primaria o menos',4:'Secundaria',5:'Media',6:'Superior'})
        edu = edu_n.where(edu_n.notna(),edu_v)
        sexo = campo('P3271').fillna(campo('P6020')).map({1:'Hombre',2:'Mujer'})
        ant = campo('P6426').mask(lambda s:s.isin([998,999]) | s.lt(0))
        hogar_keys = ['DIRECTORIO','SECUENCIA_P'] + (['HOGAR'] if 'HOGAR' in d else [])
        assert not d[hogar_keys+['ORDEN']].isna().any().any()
        # Incluir año y mes: la GEIH no se trata como panel individual.
        def id_texto(v):
            return str(int(v)) if isinstance(v,(float,np.floating,int,np.integer)) else str(v).strip()
        hogar = f'{m.anio}-{m.mes:02d}-' + d[hogar_keys].map(id_texto).agg('-'.join,axis=1)
        p = pd.DataFrame(dict(anio=m.anio,mes=m.mes,ciudad=d.AREA.map(lector.AREAS),
            id_hogar=hogar,id_persona=hogar+'-'+d.ORDEN.map(id_texto),
            ingreso=d.INGLABO.astype(float),edad=campo('P6040'),antiguedad=ant,horas=campo('P6800'),
            sexo=sexo,jefe=campo('P6050').map({1:'Si',**{k:'No' for k in range(2,10)}}),
            estudia=campo('P6170').map({1:'Si',2:'No'}),
            estado_civil=campo('P6070').where(lambda s:s.isin(range(1,7))).map(lambda x:str(int(x)) if pd.notna(x) else np.nan),
            educacion=edu))
        assert not p.id_persona.duplicated().any()
        parts.append(p)
        aud.append(dict(anio=m.anio,mes=m.mes,ocupados_fuente=n_leidos,seleccionados=len(p),
            archivos=' | '.join(str(f) for mod in lector.MODULOS for f in archivos[mod])))
        del d, raw
    data = pd.concat(parts,ignore_index=True)
    assert len(data) == int(esperado.n.sum()) == 496357
    assert not data.id_persona.duplicated().any()
    # Las categorías con ausencia estructural completa se detectan, no se imputan a ciegas.
    cobertura = data.groupby(['anio','mes'])[FEATURES].count()
    assert cobertura.gt(0).all().all(), cobertura.loc[~cobertura.gt(0).all(axis=1)]
    data.to_parquet(base,index=False)
    pd.DataFrame(aud).to_csv(out/'auditoria_base.csv',index=False)
    pd.DataFrame(disponibilidad).to_csv(out/'disponibilidad_variables.csv',index=False)
    guardar_json(dict(firma=firma,n=len(data),predictores=FEATURES,
        fuentes=estado_fuentes,python=platform.python_version()),meta)
    return data

def particionar(data, out):
    d = data.copy()
    rng = np.random.default_rng(SEED)
    asignaciones = {}
    for (anio,ciudad), g in d.groupby(['anio','ciudad'],sort=True):
        hogares = np.array(sorted(g.id_hogar.unique()))
        rng.shuffle(hogares)
        n1,n2 = int(.6*len(hogares)),int(.8*len(hogares))
        for etiqueta, ids in [('entrenamiento',hogares[:n1]),('validacion',hogares[n1:n2]),('prueba',hogares[n2:])]:
            asignaciones.update(dict.fromkeys(ids,etiqueta))
    d['conjunto'] = d.id_hogar.map(asignaciones)
    train = d.conjunto.eq('entrenamiento')
    cortes = d.loc[train].groupby(['anio','ciudad']).ingreso.quantile([.1,.5,.9]).unstack()
    cortes.columns = ['q10','q50','q90']
    cortes.reset_index().to_csv(out/'cortes_entrenamiento.csv',index=False)
    d = d.join(cortes,on=['anio','ciudad'])
    assert d[['q10','q50','q90']].notna().all().all()
    d['y'] = np.select([d.ingreso.le(d.q10),d.ingreso.le(d.q50),d.ingreso.le(d.q90)],[0,1,2],default=3).astype('int8')
    assert d.groupby('id_hogar').conjunto.nunique().max() == 1
    assert d.groupby('conjunto').y.nunique().eq(4).all()
    assert set(FEATURES).isdisjoint(['ingreso','q10','q50','q90','y','id_hogar','id_persona'])
    d[['id_persona','id_hogar','anio','mes','ciudad','conjunto','y']].to_parquet(out/'particion.parquet',index=False)
    return d

def metricas(y, pred, scores=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    r = dict(exactitud=accuracy_score(y,pred),balanceada=balanced_accuracy_score(y,pred),
        precision_macro=precision_score(y,pred,average='macro',zero_division=0),
        sensibilidad_macro=recall_score(y,pred,average='macro',zero_division=0),
        f1_macro=f1_score(y,pred,average='macro',zero_division=0),
        error_ordinal=float(np.abs(np.asarray(y)-np.asarray(pred)).mean()))
    if scores is not None:
        r['auc_macro'] = float(np.mean([roc_auc_score(np.asarray(y)==c,scores[:,c]) for c in range(4)]))
    return r

def entrenar(root, data):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix, classification_report
    from sklearn.exceptions import ConvergenceWarning
    import sklearn
    out = Path(root)/'resultados_historicos'
    d = particionar(data,out)
    del data
    subsets = {s: np.flatnonzero(d.conjunto.eq(s)) for s in ['entrenamiento','validacion','prueba']}
    it,iv,ie = [subsets[s] for s in ['entrenamiento','validacion','prueba']]
    X = d[FEATURES].copy()
    for c in CAT: X[c] = X[c].astype(object).where(X[c].notna(),np.nan)
    pre = ColumnTransformer([('num',Pipeline([('imputar',SimpleImputer(strategy='median')),('escala',StandardScaler())]),NUM),
        ('cat',Pipeline([('imputar',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)])
    zt = pre.fit_transform(X.iloc[it]).astype('float32')
    zv = pre.transform(X.iloc[iv]).astype('float32')
    # La prueba no se transforma ni evalúa durante selección.
    yt,yv = d.y.to_numpy()[it], d.y.to_numpy()[iv]
    names = list(pre.get_feature_names_out())
    joblib.dump(pre,out/'preprocesador.joblib')
    rng = np.random.default_rng(SEED)
    diag = np.sort(rng.choice(len(it),min(20000,len(it)),replace=False))
    results, grids, bests, alerts = [], [], {}, []
    def ajustar(nombre, estimator, parametros, trainz=zt, valz=zv):
        print(f'Ajustando {nombre}: {parametros}',flush=True)
        inicio=time.monotonic()
        with warnings.catch_warnings(record=True) as capturadas:
            warnings.simplefilter('always',ConvergenceWarning)
            estimator.fit(trainz,yt)
        for w in capturadas: alerts.append(dict(modelo=nombre,parametros=str(parametros),aviso=str(w.message)))
        score = estimator.predict_proba(valz) if hasattr(estimator,'predict_proba') else estimator.decision_function(valz)
        met = metricas(yv,estimator.predict(valz),score)
        met.update(modelo=nombre,parametros=str(parametros),segundos=time.monotonic()-inicio)
        grids.append(met)
        old = bests.get(nombre)
        if old is None or met['f1_macro'] > old['met']['f1_macro']:
            path = out/(nombre.replace(' ','_').replace('+','_')+'.joblib')
            joblib.dump(estimator,path)
            bests[nombre] = dict(met=met,path=path,parametros=parametros)
        del estimator
        gc.collect()
        pd.DataFrame(grids).to_csv(out/'malla_validacion.csv',index=False)
    for c in [.1,1.0]: ajustar('Logistica',LogisticRegression(C=c,max_iter=1000,solver='lbfgs'),dict(C=c))
    for profundidad in [4,8,12]: ajustar('Arbol',DecisionTreeClassifier(max_depth=profundidad,min_samples_leaf=30,random_state=SEED),dict(profundidad=profundidad))
    for mf,leaf in [('sqrt',30),(.5,30),(.5,100)]:
        ajustar('Random Forest',RandomForestClassifier(n_estimators=120,max_depth=18,min_samples_leaf=leaf,max_features=mf,n_jobs=2,random_state=SEED),dict(arboles=120,max_features=mf,min_hoja=leaf))
    for c in [.1,1.,10.]: ajustar('SVM lineal',LinearSVC(C=c,dual=False,max_iter=5000,random_state=SEED),dict(C=c))
    # ACP ajustado solo en entrenamiento; la fracción de varianza se decide en validación.
    pca = PCA(svd_solver='full').fit(zt)
    joblib.dump(pca,out/'acp.joblib')
    pt,pv = pca.transform(zt),pca.transform(zv)
    for frac in [.8,.95]:
        nc=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),frac)+1)
        for c in [.1,1.]:
            ajustar('ACP + SVM',LinearSVC(C=c,dual=False,max_iter=5000,random_state=SEED),dict(varianza=frac,componentes=nc,C=c),pt[:,:nc],pv[:,:nc])
    # Todos los registros de entrenamiento se conservan en el índice KNN.
    print('KNN: vecinos de toda la validación contra todo entrenamiento',flush=True)
    inicio = time.monotonic()
    knn = NearestNeighbors(n_neighbors=31,algorithm='kd_tree',leaf_size=60,n_jobs=2).fit(zt)
    def vecinos(z):
        return np.concatenate([knn.kneighbors(z[i:i+2000],return_distance=False) for i in range(0,len(z),2000)])
    vv = vecinos(zv)
    for k in [5,15,31]:
        prob=np.stack([(yt[vv[:,:k]]==c).mean(axis=1) for c in range(4)],axis=1)
        met=metricas(yv,prob.argmax(axis=1),prob)
        met.update(modelo='KNN',parametros=str(dict(k=k)),segundos=time.monotonic()-inicio)
        grids.append(met)
        if 'KNN' not in bests or met['f1_macro'] > bests['KNN']['met']['f1_macro']:
            bests['KNN']=dict(met=met,path=out/'KNN.joblib',parametros=dict(k=k))
    joblib.dump(dict(indice=knn,y=yt,k=bests['KNN']['parametros']['k']),out/'KNN.joblib')
    del vv,knn
    def scores_model(nombre,z):
        model=joblib.load(bests[nombre]['path'])
        if nombre=='KNN':
            v=np.concatenate([model['indice'].kneighbors(z[i:i+2000],return_distance=False)[:,:model['k']] for i in range(0,len(z),2000)])
            score=np.stack([(model['y'][v]==c).mean(axis=1) for c in range(4)],axis=1)
        else:
            if nombre=='ACP + SVM': z=pca.transform(z)[:,:bests[nombre]['parametros']['componentes']]
            score=model.predict_proba(z) if hasattr(model,'predict_proba') else model.decision_function(z)
        del model
        return score
    for nombre, item in bests.items():
        print('Diagnóstico de entrenamiento:',nombre,flush=True)
        score=scores_model(nombre,zt[diag])
        results.append(dict(modelo=nombre,conjunto='Entrenamiento (20 mil)',**metricas(yt[diag],score.argmax(axis=1),score)))
        results.append(dict(modelo=nombre,conjunto='Validacion',**{k:v for k,v in item['met'].items() if k not in ['modelo','parametros','segundos']}))
    mayor=int(np.bincount(yt,minlength=4).argmax())
    baseline=metricas(yv,np.full(len(yv),mayor))
    pd.DataFrame(grids).to_csv(out/'malla_validacion.csv',index=False)
    pd.DataFrame(results).to_csv(out/'comparacion.csv',index=False)
    ganador=max(bests,key=lambda n:bests[n]['met']['f1_macro'])
    print('Modelo seleccionado en validación:',ganador,flush=True)
    # Costos multiclase: mantener el escenario ordinal separado de la selección por F1.
    # No se convierten márgenes SVM en probabilidades sin calibrar.
    prob_names=[n for n in bests if n in ['Logistica','Arbol','Random Forest','KNN']]
    probabilistico=max(prob_names,key=lambda n:bests[n]['met']['f1_macro'])
    proba=scores_model(probabilistico,zv)
    costrows=[]
    # Filas = clase real; columnas = clase predicha; sobreestimar vulnerabilidad
    # no se interpreta como costo real. Escenario fijo penaliza infraclasificar el ingreso.
    cidx=np.arange(4)
    real,pred=np.meshgrid(cidx,cidx,indexing='ij')
    costo=np.where(pred<real,2*(real-pred),pred-real).astype(float)
    for factor in [1.,1.5,2.,3.]:
        matriz=np.where(pred<real,factor*(real-pred),pred-real)
        decision=(proba @ matriz).argmin(axis=1)
        costrows.append(dict(factor=factor,costo_por_mil=float(costo[yv,decision].mean()*1000),**metricas(yv,decision)))
    factor=min(costrows,key=lambda r:r['costo_por_mil'])['factor']
    pd.DataFrame(costrows).to_csv(out/'costos_validacion.csv',index=False)
    # Importancia por permutación del bosque: 5000 casos fijos de validación,
    # tres permutaciones por predictor original, sin tocar prueba.
    print('Importancia por permutación (validación)',flush=True)
    rf=joblib.load(bests['Random Forest']['path'])
    impidx=rng.choice(len(iv),min(5000,len(iv)),replace=False)
    xi=X.iloc[iv[impidx]].copy(); yi=yv[impidx]
    from sklearn.metrics import f1_score
    refscore=f1_score(yi,rf.predict(pre.transform(xi)),average='macro')
    importance=[]
    for c in FEATURES:
        values=[]
        for rep in range(3):
            xx=xi.copy(); xx[c]=rng.permutation(xx[c].to_numpy())
            values.append(refscore-f1_score(yi,rf.predict(pre.transform(xx)),average='macro'))
        prefix=('num__'+c) if c in NUM else ('cat__'+c+'_')
        positions=[j for j,n in enumerate(names) if n==prefix or n.startswith(prefix)]
        importance.append(dict(variable=c,permutacion=np.mean(values),sd=np.std(values),impureza=float(rf.feature_importances_[positions].sum())))
    pd.DataFrame(importance).sort_values('permutacion',ascending=False).to_csv(out/'importancias.csv',index=False)
    del rf
    # K-means es no supervisado y nunca recibe y ni ingreso.
    print('K-means: selección k=2,...,6',flush=True)
    clrows=[]; clbest=None
    silidx=rng.choice(len(zv),min(2000,len(zv)),replace=False)
    for k in [2,3,4,5,6]:
        km=KMeans(n_clusters=k,n_init=3,max_iter=150,random_state=SEED,algorithm='lloyd').fit(zt)
        lab=km.predict(zv[silidx])
        sil=silhouette_score(zv[silidx],lab) if len(np.unique(lab))>1 else np.nan
        clrows.append(dict(k=k,inercia=km.inertia_,silhouette_validacion=sil,iteraciones=km.n_iter_))
        if clbest is None or sil>clbest[0]:
            clbest=(sil,k); joblib.dump(km,out/'kmeans.joblib')
        del km
    pd.DataFrame(clrows).to_csv(out/'clusters_validacion.csv',index=False)
    km=joblib.load(out/'kmeans.joblib')
    km2=KMeans(n_clusters=clbest[1],n_init=3,max_iter=150,random_state=SEED+1).fit(zt)
    estabilidad=adjusted_rand_score(km.predict(zv),km2.predict(zv))
    del km2
    # Congelar elecciones ANTES de transformar y evaluar prueba.
    config=dict(ganador=ganador,probabilistico=probabilistico,factor_coste=factor,k_clusters=clbest[1],
        parametros={n:v['parametros'] for n,v in bests.items()},semilla=SEED)
    guardar_json(config,out/'decisiones_congeladas.json')
    print('Evaluación final en prueba reservada',flush=True)
    ze=pre.transform(X.iloc[ie]).astype('float32'); ye=d.y.to_numpy()[ie]
    score=scores_model(ganador,ze); yp=score.argmax(axis=1)
    final=metricas(ye,yp,score)
    cm=confusion_matrix(ye,yp,labels=np.arange(4))
    pd.DataFrame(cm,index=CLASES,columns=CLASES).to_csv(out/'confusion_prueba.csv')
    guardar_json(classification_report(ye,yp,labels=np.arange(4),target_names=CLASES,output_dict=True,zero_division=0),out/'reporte_clases.json')
    # Bootstrap por hogar-mes, modelo y cortes fijos, no diseño muestral GEIH.
    ids=pd.factorize(d.iloc[ie].id_hogar)[0]; ng=ids.max()+1
    blocks=np.zeros((ng,16),dtype=np.int32)
    np.add.at(blocks,(ids,ye*4+yp),1)
    def metcm(mat):
        true=mat.sum(axis=1); predicted=mat.sum(axis=0); diagv=np.diag(mat)
        rec=np.divide(diagv,true,out=np.zeros(4),where=true>0)
        prec=np.divide(diagv,predicted,out=np.zeros(4),where=predicted>0)
        f=np.divide(2*prec*rec,prec+rec,out=np.zeros(4),where=prec+rec>0)
        return [diagv.sum()/mat.sum(),rec.mean(),f.mean()]
    boot=[]
    for b in range(300):
        mat=blocks[rng.integers(0,ng,ng)].sum(axis=0).reshape(4,4)
        boot.append(metcm(mat))
    cis=dict(zip(['exactitud','balanceada','f1_macro'],np.quantile(boot,[.025,.975],axis=0).T.tolist()))
    pp=scores_model(probabilistico,ze)
    matriz=np.where(pred<real,factor*(real-pred),pred-real)
    decis=(pp @ matriz).argmin(axis=1)
    costo_final=float(costo[ye,decis].mean()*1000)
    costo_argmax=float(costo[ye,pp.argmax(axis=1)].mean()*1000)
    # Regla constante elegida con frecuencias de entrenamiento, no con prueba.
    constante=int((np.bincount(yt,minlength=4)/len(yt) @ costo).argmin())
    costo_const=float(costo[ye,constante].mean()*1000)
    lab=km.predict(ze)
    perfil=d.iloc[ie][NUM+['ciudad','educacion','ingreso','y']].copy()
    perfil['cluster']=lab+1
    resumen_cluster=perfil.groupby('cluster').agg(n=('y','size'),edad=('edad','mean'),horas=('horas','mean'),
        antiguedad=('antiguedad','median'),ingreso_mediano=('ingreso','median'),clase_media=('y','mean'))
    resumen_cluster.to_csv(out/'perfiles_clusters_prueba.csv')
    pd.crosstab(perfil.cluster,perfil.y,normalize='index').to_csv(out/'clusters_clases_prueba.csv')
    predicciones=d.iloc[ie][['id_persona','id_hogar','anio','mes','ciudad','y']].copy()
    predicciones['prediccion']=yp; predicciones['cluster']=lab+1
    predicciones.to_parquet(out/'predicciones_prueba.parquet',index=False)
    grupos=[]
    for tipo in ['ciudad','anio']:
        for valor,g in predicciones.groupby(tipo):
            grupos.append(dict(tipo=tipo,grupo=valor,n=len(g),**metricas(g.y,g.prediccion)))
    pd.DataFrame(grupos).to_csv(out/'desempeno_subgrupos.csv',index=False)
    # Archivos para figuras, con 5000 puntos como máximo solo en dispersión.
    pd.DataFrame(dict(componente=np.arange(1,len(names)+1),varianza=pca.explained_variance_ratio_,
        acumulada=np.cumsum(pca.explained_variance_ratio_))).to_csv(out/'acp_varianza.csv',index=False)
    pd.DataFrame(pca.components_[:3].T,index=names,columns=['CP1','CP2','CP3']).to_csv(out/'acp_cargas.csv')
    iz=rng.choice(len(ze),min(5000,len(ze)),replace=False)
    coords=pca.transform(ze[iz])[:,:2]
    pd.DataFrame(dict(CP1=coords[:,0],CP2=coords[:,1],cluster=lab[iz]+1,y=ye[iz])).to_csv(out/'acp_dispersion.csv',index=False)
    pd.DataFrame(alerts,columns=['modelo','parametros','aviso']).to_csv(out/'avisos_ajuste.csv',index=False)
    resumen=dict(n=len(d),n_tr=len(it),n_va=len(iv),n_te=len(ie),n_hogares=d.id_hogar.nunique(),
        n_pred=len(FEATURES),n_codificados=len(names),features=FEATURES,ganador=ganador,
        parametros=config['parametros'],validacion=bests[ganador]['met'],prueba=final,ic=cis,
        baseline_validacion=baseline,baseline_prueba=metricas(ye,np.full(len(ye),mayor)),
        costos=dict(modelo=probabilistico,factor=factor,prueba=costo_final,argmax=costo_argmax,constante=costo_const,clase_constante=constante),
        clustering=dict(k=clbest[1],silhouette_validacion=clbest[0],estabilidad_ari=estabilidad),
        avisos=len(alerts),python=platform.python_version(),sklearn=sklearn.__version__)
    guardar_json(resumen,out/'resumen.json')
    pd.crosstab(d.conjunto,d.y).to_csv(out/'distribucion_clases.csv')
    d.groupby('anio')[FEATURES].agg(lambda s:s.isna().sum()).to_csv(out/'faltantes_por_anio.csv')
    print(json.dumps(resumen,ensure_ascii=False,indent=2),flush=True)
    return resumen

def ejecutar(root, recalcular=False):
    root=Path(root)
    with threadpool_limits(limits=2):
        data=construir(root)
        out=root/'resultados_historicos'
        firma=hashlib.sha256(Path(__file__).read_bytes()+(out/'base_manifest.json').read_bytes()).hexdigest()
        meta=out/'modelos_manifest.json'
        if not recalcular and meta.exists() and json.loads(meta.read_text())['firma']==firma:
            print('Modelos: reutilizando resultados de la misma base y código',flush=True)
            return json.loads((out/'resumen.json').read_text(encoding='utf-8'))
        resumen=entrenar(root,data)
        guardar_json(dict(firma=firma),meta)
        return resumen

if __name__=='__main__':
    import sys
    ejecutar(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent)
