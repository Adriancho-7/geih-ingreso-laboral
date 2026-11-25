"""Coeficientes de modelos temporales de mediana, con ajuste final hasta 2025.

No selecciona modelos: solo documenta estimaciones de la malla ya evaluada.
"""
from pathlib import Path
import json,sys,hashlib,warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits
from series_temporales import AREAS,SAR,features,SEED
threadpool_limits(1)

def run(root):
    root=Path(root);out=root/'resultados_series_final';source=root/'resultados_series/series_con_origen.csv';signature=hashlib.sha256(Path(__file__).read_bytes()+source.read_bytes()).hexdigest()
    manifest=out/'parametros_mediana_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['digest']==signature:return
    data=pd.read_csv(source,parse_dates=['fecha']);rows=[];alerts=[]
    for area in AREAS:
        print('Coeficientes temporales',area,flush=True)
        y=data[(data.ciudad==area)&data.indicador.eq('mediana')].set_index('fecha').usado.dropna().asfreq('MS');z=np.log(y)
        def add(name,params,se=None,p=None):
            for key,val in params.items():
                for j,number in enumerate(np.asarray(val).ravel()):
                    if number is None:continue
                    try:number=float(number)
                    except (TypeError,ValueError):continue
                    if not np.isfinite(number):continue
                    rows.append(dict(ciudad=area,modelo=name,parametro=key if np.asarray(val).size==1 else f'{key}[{j}]',estimacion=float(number),se=float(se[key]) if se is not None and key in se else np.nan,p=float(p[key]) if p is not None and key in p else np.nan))
        add('Ingenuo',{'ultimo_log':z.iloc[-1]});add('Deriva',{'ultimo_log':z.iloc[-1],'pendiente_log':(z.iloc[-1]-z.iloc[0])/(len(z)-1)});add('Estacional',{'ultimos12_log':z.iloc[-12:].to_numpy()})
        for name in ['SES','Holt','Holt amortiguado','ETS AAA','ETS AAdA']:
            m=ExponentialSmoothing(z,trend=None if name=='SES' else 'add',damped_trend=name in ['Holt amortiguado','ETS AAdA'],seasonal='add' if name.startswith('ETS') else None,seasonal_periods=12 if name.startswith('ETS') else None,initialization_method='estimated').fit();add(name,m.params)
        estimated={}
        for name,(order,season) in SAR.items():
            ex=np.column_stack([(z.index>=pd.Timestamp('2022-01')).astype(float),((z.index>=pd.Timestamp('2020-03'))&(z.index<=pd.Timestamp('2020-07'))).astype(float)]) if name.startswith('SARIMAX') else None
            with warnings.catch_warnings(record=True) as ws:
                m=SARIMAX(z,order=order,seasonal_order=season,exog=ex,enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
            alerts.extend(dict(ciudad=area,modelo=name,aviso=str(w.message)) for w in ws);estimated[name]=m;add(name,m.params,m.bse,m.pvalues)
        for D in [0,1]:
            possible=[(name,m) for name,m in estimated.items() if not name.startswith('SARIMAX') and SAR[name][1][1]==D and m.mle_retvals['converged']]
            if possible:
                name,m=min(possible,key=lambda t:t[1].aicc);add(f'Auto-malla D{D}',m.params,m.bse,m.pvalues)
        from prophet import Prophet
        for scale in [.05,.20]:
            m=Prophet(changepoint_prior_scale=scale,yearly_seasonality=4,weekly_seasonality=False,daily_seasonality=False,uncertainty_samples=0);m.fit(pd.DataFrame({'ds':z.index,'y':z.values}),seed=SEED);add(f'Prophet {scale:.2f}',m.params)
        from arch import arch_model
        for name in ['AR1-ARCH1','AR1-GARCH11']:
            m=arch_model(100*z.diff().dropna(),mean='AR',lags=1,vol='ARCH' if name=='AR1-ARCH1' else 'GARCH',p=1,q=0 if name=='AR1-ARCH1' else 1,rescale=False).fit(disp='off',show_warning=False);add(name,m.params,m.std_err,m.pvalues)
        cols=json.loads((root/'resultados_series'/f'{AREAS.index(area)}_mediana/regresion_elegida.json').read_text(encoding='utf8'))['variables'];X=np.array([[1]+[features(z.values[:i],z.index[i])[c] for c in cols] for i in range(12,len(z))]);m=sm.OLS(z.values[12:],X).fit(cov_type='HAC',cov_kwds={'maxlags':12});keys=['intercepto',*cols];add('Regresion exhaustiva',dict(zip(keys,m.params)),dict(zip(keys,m.bse)),dict(zip(keys,m.pvalues)))
        pd.DataFrame(rows).to_csv(out/'parametros_mediana_todos.csv',index=False)
    pd.DataFrame(alerts).to_csv(out/'avisos_parametros.csv',index=False);manifest.write_text(json.dumps({'digest':signature}),encoding='utf8')

if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else Path.cwd())
