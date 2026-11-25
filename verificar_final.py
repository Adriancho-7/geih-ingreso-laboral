"""Verificación independiente de integridad, métricas, regla de costos y separación de datos."""
from pathlib import Path
import json, sys, math, tempfile
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from ml_final import AREAS, GRIDS, FINAL, CIUDADES, cargar, epoca_restaurada, features_de, inputs, scores, NEIGHBOR_CACHE
import decision_costos as dc
from series_temporales import AREAS as TAREAS


def etiquetas(frame, cortes):
    q = cortes.reindex(pd.MultiIndex.from_arrays([frame.ciudad.to_numpy(dtype=object), frame.anio.to_numpy()])).to_numpy()
    assert np.isfinite(q).all()
    return np.sum(frame.ingreso.to_numpy()[:, None] > q, axis=1)


def run(root):
    root = Path(root); checks = []
    monthly = pd.read_csv(root/'datos_historicos/mensual.csv')
    for i, a in enumerate(AREAS):
        p = root/'resultados_ml_final'/str(i); s = json.loads((p/'resumen.json').read_text(encoding='utf8'))
        d = cargar(root, a)
        groups = {k: set(v.id_hogar) for k, v in d.groupby('conjunto')}
        assert groups['entrenamiento'].isdisjoint(groups['validacion']) and groups['entrenamiento'].isdisjoint(groups['prueba']) and groups['validacion'].isdisjoint(groups['prueba'])
        computed = d.groupby(['anio', 'mes']).ingreso.agg(n='size', p10=lambda v: v.quantile(.1), mediana='median', p90=lambda v: v.quantile(.9), promedio='mean')
        published = monthly[monthly.ciudad == a].set_index(['anio', 'mes']).reindex(computed.index)
        assert np.allclose(computed[['n', 'p10', 'mediana', 'p90', 'promedio']], published[['n', 'p10', 'mediana', 'p90', 'promedio']], rtol=1e-10, atol=1e-6), ('Microdatos/serie no coinciden', a)
        folds = pd.read_parquet(p/'pliegues.parquet').merge(d[['id_persona', 'id_hogar', 'conjunto']], on='id_persona', validate='one_to_one')
        assert set(folds.conjunto) == {'entrenamiento'} and set(folds.pliegue) == {1, 2, 3}
        assert folds.groupby('id_hogar').pliegue.nunique().max() == 1
        assert len(folds) == s['entrenamiento'] and len(d) == s['n']
        pred = pd.read_parquet(p/'predicciones.parquet'); metrics = pd.read_csv(p/'prueba.csv'); cm = pd.read_csv(p/'confusiones.csv'); val = pd.read_csv(p/'validacion.csv')
        curve = pd.read_csv(p/'red_aprendizaje.csv').to_dict('records'); meta = json.loads((p/'parametros_6.json').read_text(encoding='utf8')); assert meta['epoca'] == epoca_restaurada(curve)
        neural = joblib.load(p/'modelo_6.joblib'); assert neural['model'].t_/neural['n_ajuste'] == meta['epoca']; del neural
        # Ganador: familia probabilística de menor costo esperado en validación (desempate F1).
        winner = val[val.modelo.isin(dc.PROBABILISTICOS)].sort_values(['costo_regla', 'f1_macro', 'modelo'], ascending=[True, False, True]).modelo.iloc[0]
        assert winner == s['ganador'] and metrics.loc[metrics.ganador.astype(bool), 'modelo'].item() == winner
        # Cortes por ciudad y año: recalculados solo con entrenamiento.
        cortes = pd.read_csv(p/'cortes.csv').set_index(['ciudad', 'anio'])
        tr = d[d.conjunto == 'entrenamiento']; again = tr.groupby(['ciudad', 'anio']).ingreso.quantile([.1, .5, .9]).unstack(); again.columns = ['q10', 'q50', 'q90']
        assert np.allclose(again.reindex(cortes.index).to_numpy(), cortes.to_numpy())
        test = d[d.conjunto == 'prueba'].set_index('id_persona'); expected = pd.Series(etiquetas(test, cortes), index=test.index)
        for model in GRIDS:
            v = pred[pred.modelo == model]; assert len(v) == s['prueba']; assert np.array_equal(v.real.to_numpy(), expected.reindex(v.id_persona).to_numpy())
            row = metrics[metrics.modelo == model].iloc[0]
            assert np.isclose(f1_score(v.real, v.predicha, average='macro', zero_division=0), row.f1_macro)
            saved = cm[(cm.modelo == model) & (cm.regla == 'argmax')].pivot(index='real', columns='predicha', values='n').to_numpy()
            assert np.array_equal(confusion_matrix(v.real, v.predicha, labels=range(4)), saved)
            if model in dc.PROBABILISTICOS:
                assert np.isclose(dc.costo_medio(v.real, v.predicha_regla), row.costo_regla)
                saved = cm[(cm.modelo == model) & (cm.regla == 'costos')].pivot(index='real', columns='predicha', values='n').to_numpy()
                assert np.array_equal(confusion_matrix(v.real, v.predicha_regla, labels=range(4)), saved)
            else:
                assert (v.predicha_regla == -1).all()
        # La regla de costos se reproduce desde el modelo guardado: modelo -> probabilidades -> decisión.
        pre = joblib.load(p/'preprocesador.joblib'); bundle = joblib.load(p/f'modelo_{list(GRIDS).index(winner)}.joblib')
        NEIGHBOR_CACHE.clear(); te_ord = test.loc[pred[pred.modelo == winner].id_persona]
        sc = scores(bundle, pre.transform(inputs(te_ord, features_de(a))).astype('float32')); del bundle
        w = pred[pred.modelo == winner].set_index('id_persona')
        assert np.array_equal(dc.decidir(sc), w.predicha_regla.to_numpy())
        prob = pd.read_parquet(p/'probabilidades_ganador.parquet').set_index('id_persona').reindex(w.index)
        assert np.allclose(prob.to_numpy(), sc, atol=1e-6)
        clara = np.abs(sc[:, 0]-dc.TAU) > 1e-9
        assert np.array_equal((sc[:, 0] >= dc.TAU)[clara], (w.predicha_regla.to_numpy() == 0)[clara])
        if a == FINAL:
            # Comparación pareada: en cada área, mismas personas de prueba y mismas clases que su modelo.
            for j, city in enumerate(CIUDADES):
                cp = pd.read_parquet(root/'resultados_ml_final'/str(j)/'predicciones.parquet'); cp = cp[cp.modelo == 'Logistica']
                mine = w.reset_index().merge(test[['ciudad']].reset_index(), on='id_persona'); mine = mine[mine.ciudad == city]
                assert set(mine.id_persona) == set(cp.id_persona)
                assert np.array_equal(mine.set_index('id_persona').real.reindex(cp.id_persona).to_numpy(), cp.real.to_numpy())
            from cargar_y_predecir import predecir
            sample = test.head(5); got = predecir(sample.reset_index(), root)
            assert np.array_equal(got.clase_asignada.to_numpy(), w.reindex(sample.index).predicha_regla.to_numpy()+1)
        del d, test, tr
        # Regresión: partición propia (histórica en áreas; aleatoria por hogar-mes y año en la nacional).
        if a == FINAL:
            from modelado_historico import particionar
            base = pd.read_parquet(root/'datos_ml_nacional/base_individual.parquet')
            with tempfile.TemporaryDirectory() as temp: rd = particionar(base, Path(temp))
            del base
        else:
            rd = pd.read_parquet(root/'resultados_historicos/base_individual.parquet'); rd = rd[rd.ciudad == a].merge(pd.read_parquet(root/'resultados_historicos/particion.parquet')[['id_persona', 'conjunto']], on='id_persona', validate='one_to_one')
        combos = pd.read_csv(root/'resultados_regresion_final'/str(i)/'combinaciones.csv'); assert len(combos) == 511 and combos.variables.nunique() == 511
        for k in range(1, 10): assert sum(combos.k == k) == math.comb(9, k)
        rp = root/'resultados_regresion_final'/str(i); bundle = joblib.load(rp/'modelos.joblib'); rt = rd[rd.conjunto == 'prueba']; blocks = []
        for var in bundle['variables']:
            im, proc = bundle['preprocesamiento'][var]; arr = rt[[var]].astype(float) if var in ['edad', 'antiguedad', 'horas', 'anio'] else rt[[var]].astype(object).where(rt[[var]].notna(), np.nan); blocks.append(proc.transform(im.transform(arr)))
        X = np.column_stack([np.ones(len(rt)), *blocks]); regmet = pd.read_csv(rp/'prueba.csv'); regval = pd.read_csv(rp/'regularizacion.csv')
        for model in ['OLS subconjunto', 'Ridge', 'Lasso']:
            if model == 'OLS subconjunto': forecast = X[:, bundle['columnas_OLS']] @ bundle['coef_OLS']
            else:
                alpha = regval[regval.modelo == model].sort_values('RMSE').iloc[0].alpha; forecast = bundle['modelos_regularizados'][f'{model}_{alpha:g}'].predict(X[:, 1:])
            row = regmet[regmet.modelo == model].iloc[0]
            assert np.isclose(np.sqrt(mean_squared_error(rt.ingreso, forecast)), row.RMSE)
            assert np.isclose(mean_absolute_error(rt.ingreso, forecast), row.MAE) and np.isclose(r2_score(rt.ingreso, forecast), row.R2)
        del X, blocks, rt, bundle, rd
        checks.append(dict(dominio=a, n=s['n'], pliegues=3, clasificadores=7, regresiones=511, ganador=winner, verificado=True))
    assert (root/'modelo_final/modelo.joblib').exists() and (root/'modelo_final/FICHA_MODELO.md').exists()
    data = pd.read_csv(root/'resultados_series/series_con_origen.csv'); filled = data[data.origen == 'nacional_sustituto']; assert len(filled) == 100 and filled.fecha.str[:7].isin(['2020-03', '2020-04', '2020-05', '2020-06', '2020-07']).all()
    nation = data[data.ciudad == 'Nacional'].set_index(['fecha', 'indicador']).observado
    assert np.allclose(filled.usado, nation.reindex(pd.MultiIndex.from_frame(filled[['fecha', 'indicador']])).to_numpy())
    comp = pd.read_csv(root/'resultados_series_final/comparacion.csv'); sel = comp[comp.ganador_final]; assert len(sel) == 24 and not sel.modelo.str.startswith('AR1-').any()
    for _, v in sel.iterrows():
        candidate = comp[(comp.ciudad == v.ciudad) & (comp.indicador == v.indicador) & comp.seleccionable].sort_values(['RMSE_validacion', 'modelo']).iloc[0]; assert candidate.modelo == v.modelo
        p = pd.read_csv(root/'resultados_series'/f'{TAREAS.index(v.ciudad)}_{v.indicador}/predicciones.csv'); p = p[(p.modelo == v.modelo) & (p.etapa == 'prueba')]; assert len(p) == 12; assert np.isclose(np.sqrt(np.mean((p.observado-p.pred)**2)), v.RMSE)
    f = pd.read_csv(root/'resultados_series_final/intervalos.csv'); assert len(f) == 576 and np.isfinite(f[['pred', 'lo80', 'hi80', 'lo95', 'hi95']]).all().all()
    assert (f.lo95 <= f.lo80).all() and (f.lo80 <= f.hi80).all() and (f.hi80 <= f.hi95).all() and (f.lo95 > 0).all()
    cf = pd.read_csv(root/'resultados_series_final/contrafactuales.csv'); assert len(cf) == 528 and cf.loc[cf.sustituido, 'desviacion_observada'].isna().all()
    target = root/'verificacion_final.json'; target.write_text(json.dumps(dict(verificado=True, modelo_final=FINAL, dominios=checks, series=24, pronosticos_intervalos=len(f), sustituciones=len(filled)), ensure_ascii=False, indent=2), encoding='utf8')
    print('VERIFICACIÓN COMPLETA: modelo final nacional + 5 áreas de comparación, 42 clasificadores, regla de costos, 3066 regresiones, 24 series e intervalos.')


if __name__ == '__main__': run(sys.argv[1] if len(sys.argv) > 1 else Path.cwd())
