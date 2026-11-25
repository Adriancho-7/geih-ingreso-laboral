"""GEIH: reconstrucción, intervención y pronóstico con validación temporal.

No descarga ni modifica los microdatos; conserva los faltantes originales.
Resultados por serie con caché vinculada al código y a la fuente.
"""
from pathlib import Path
import os, sys, json, hashlib, itertools, warnings, logging, time
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera
from threadpoolctl import threadpool_limits
threadpool_limits(2)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
SEED=78018
IND=['p10','mediana','p90','promedio']
AREAS=['Nacional','Medellín AM','Barranquilla AM','Bogotá','Bucaramanga AM','Cali AM']
VARS=['lag1','lag2','lag12','tendencia','seno','coseno','marco2022','pulso2020']
SUBSETS=[tuple(c) for k in range(1,9) for c in itertools.combinations(VARS,k)]
REG_CACHE={}
FIT_CACHE={}
SAR={
 'SARIMA011-011':((0,1,1),(0,1,1,12)),
 'SARIMA210-011':((2,1,0),(0,1,1,12)),
 'SARIMA111-011':((1,1,1),(0,1,1,12)),
 'SARIMA011-100':((0,1,1),(1,0,0,12)),
 'SARIMA210-100':((2,1,0),(1,0,0,12)),
 'SARIMA111-100':((1,1,1),(1,0,0,12)),
 'SARIMAX011-011':((0,1,1),(0,1,1,12)),
 'SARIMAX011-100':((0,1,1),(1,0,0,12))}
MODELS=['Ingenuo','Estacional','Deriva','SES','Holt','Holt amortiguado','ETS AAA','ETS AAdA',*SAR,
        'Auto-malla D0','Auto-malla D1',
        'Prophet 0.05','Prophet 0.20','LSTM 80','LSTM 160','AR1-ARCH1','AR1-GARCH11','Regresion exhaustiva']

def savejson(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:float(x)),encoding='utf-8')

def features(history,date):
    t=(date.year-2008)*12+date.month-7
    return dict(lag1=history[-1],lag2=history[-2],lag12=history[-12],tendencia=t/12,
                seno=np.sin(2*np.pi*date.month/12),coseno=np.cos(2*np.pi*date.month/12),
                marco2022=float(date>=pd.Timestamp('2022-01-01')),
                pulso2020=float(pd.Timestamp('2020-03-01')<=date<=pd.Timestamp('2020-07-01')))

def regfit(y,cols,h=12):
    z=np.log(y.to_numpy()); dates=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=h,freq='MS')
    cachekey=(str(y.index[0]),tuple(z))
    if cachekey not in REG_CACHE:
        rows=[]
        for i in range(12,len(z)):
            row=features(z[:i],y.index[i]);rows.append([1]+[row[v] for v in VARS])
        REG_CACHE[cachekey]=np.asarray(rows)
    X=REG_CACHE[cachekey][:,[0]+[VARS.index(v)+1 for v in cols]]
    b=np.linalg.lstsq(X,z[12:],rcond=None)[0]
    hist=list(z)
    for date in dates:
        row=features(hist,date); hist.append(float(np.dot([1]+[row[v] for v in cols],b)))
    return np.exp(hist[-h:]), z[12:]-X@b, {'coeficientes':dict(zip(['intercepto',*cols],b)),
             'rango':int(np.linalg.matrix_rank(X)), 'columnas':len(cols)+1}

def fitmodel(y,name,h=12,cols=None):
    key=(name,h,tuple(np.asarray(y)),str(y.index[0]),tuple(cols or []))
    if key in FIT_CACHE:return FIT_CACHE[key]
    result=_fitmodel(y,name,h,cols)
    FIT_CACHE[key]=result
    return result

def _fitmodel(y,name,h=12,cols=None):
    z=np.log(y); n=len(z); fut=pd.date_range(z.index[-1]+pd.offsets.MonthBegin(),periods=h,freq='MS')
    meta={}; residual=None
    if name.startswith('Auto-malla'):
        D=int(name[-1]);candidates=[]
        for candidate,(order,season) in SAR.items():
            if not candidate.startswith('SARIMAX') and season[1]==D:
                cp,cr,cm=fitmodel(y,candidate,h,cols)
                if cm.get('convergio',True): candidates.append((cm['aicc'],candidate,cp,cr,cm))
        if not candidates:raise ValueError('Ningún modelo de la malla convergió')
        _,chosen,pred,resid,meta=min(candidates,key=lambda a:a[0])
        return pred,resid,dict(meta,seleccion_AICc=chosen)
    if name=='Ingenuo':
        p=np.repeat(z.iloc[-1],h); residual=z.diff().dropna().values
    elif name=='Estacional':
        p=np.resize(z.iloc[-12:].values,h); residual=z.diff(12).dropna().values
    elif name=='Deriva':
        slope=(z.iloc[-1]-z.iloc[0])/(n-1); p=z.iloc[-1]+np.arange(1,h+1)*slope
        residual=z.diff().dropna().values-slope
    elif name in ['SES','Holt','Holt amortiguado','ETS AAA','ETS AAdA']:
        m=ExponentialSmoothing(z,trend=None if name=='SES' else 'add',
            damped_trend=name in ['Holt amortiguado','ETS AAdA'],
            seasonal='add' if name.startswith('ETS') else None,
            seasonal_periods=12 if name.startswith('ETS') else None,initialization_method='estimated').fit()
        p=m.forecast(h).values; residual=np.asarray(m.resid)[12:]
        meta={'aic':m.aic,'aicc':m.aicc,'bic':m.bic,'convergio':bool(m.mle_retvals.success)}
    elif name in SAR:
        order,season=SAR[name]; isx=name.startswith('SARIMAX')
        def ex(idx): return np.column_stack([(idx>=pd.Timestamp('2022-01-01')).astype(float),
              ((idx>=pd.Timestamp('2020-03-01'))&(idx<=pd.Timestamp('2020-07-01'))).astype(float)])
        m=SARIMAX(z,order=order,seasonal_order=season,exog=ex(z.index) if isx else None,
                  enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
        p=m.forecast(h,exog=ex(fut) if isx else None).values
        residual=np.asarray(m.resid)[max(13,m.loglikelihood_burn):]
        meta={'aic':m.aic,'aicc':m.aicc,'bic':m.bic,'D':season[1], 'convergio':bool(m.mle_retvals['converged'])}
    elif name.startswith('Prophet'):
        from prophet import Prophet
        m=Prophet(changepoint_prior_scale=float(name.split()[1]),yearly_seasonality=4,
                  weekly_seasonality=False,daily_seasonality=False,uncertainty_samples=0)
        frame=pd.DataFrame({'ds':z.index,'y':z.values})
        m.fit(frame,seed=SEED); p=m.predict(pd.DataFrame({'ds':fut})).yhat.values
        residual=z.values-m.predict(frame).yhat.values
    elif name.startswith('LSTM'):
        import torch
        torch.set_num_threads(2); torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
        dz=np.diff(z.values); mu=dz.mean(); sd=max(dz.std(),1e-6); u=(dz-mu)/sd; w=12
        X=torch.tensor(np.array([u[i-w:i] for i in range(w,len(u))]),dtype=torch.float32).unsqueeze(-1)
        target=torch.tensor(u[w:],dtype=torch.float32).unsqueeze(-1)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.rnn=torch.nn.LSTM(1,16,batch_first=True); self.out=torch.nn.Linear(16,1)
            def forward(self,x): return self.out(self.rnn(x)[0][:,-1,:])
        model=Net(); opt=torch.optim.Adam(model.parameters(),lr=.01); losses=[]
        for epoch in range(int(name.split()[1])):
            opt.zero_grad(); loss=((model(X)-target)**2).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); losses.append(float(loss.detach()))
        model.eval(); hist=list(u); pp=[]
        with torch.no_grad():
            residual=(target-model(X)).numpy().ravel()*sd
            for _ in range(h):
                q=float(model(torch.tensor(hist[-w:],dtype=torch.float32).reshape(1,w,1))[0,0]); hist.append(q);pp.append(q*sd+mu)
        p=z.iloc[-1]+np.cumsum(pp); meta={'perdidas':losses,'ventana':w,'neuronas':16,'escala_media':mu,'escala_sd':sd}
    elif name.startswith('AR1-'):
        from arch import arch_model
        changes=100*z.diff().dropna()
        m=arch_model(changes,mean='AR',lags=1,vol='ARCH' if name=='AR1-ARCH1' else 'GARCH',
                     p=1,q=0 if name=='AR1-ARCH1' else 1,rescale=False).fit(disp='off',show_warning=False)
        fc=m.forecast(horizon=h,reindex=False)
        p=z.iloc[-1]+np.cumsum(fc.mean.iloc[-1].values)/100
        residual=m.resid.dropna().values/100
        meta={'aic':m.aic,'bic':m.bic,'convergio':m.convergence_flag==0,
              'sigma':m.conditional_volatility.dropna().tolist(), 'sigma_futura':np.sqrt(fc.residual_variance.iloc[-1]).tolist()}
    elif name=='Regresion exhaustiva': return regfit(y,cols,h)
    else: raise ValueError(name)
    return np.exp(p),np.asarray(residual),meta

def metrics(actual,pred,train):
    a=np.asarray(actual);p=np.asarray(pred);err=a-p
    scale=np.mean(np.abs(np.asarray(train)[12:]-np.asarray(train)[:-12]))
    return dict(RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(np.abs(err))),
                MAPE=float(100*np.mean(np.abs(err/a))),MASE=float(np.mean(np.abs(err))/scale))

def diagnostics(resid):
    r=np.asarray(resid);r=r[np.isfinite(r)]; lb=acorr_ljungbox(r,lags=[12,24],return_df=True)
    return {'n_resid':len(r),'LB12':float(lb.lb_pvalue.iloc[0]),'LB24':float(lb.lb_pvalue.iloc[1]),
            'JB':float(jarque_bera(r).pvalue),'ARCH12':float(het_arch(r,nlags=12)[1])}

def load_series(root,out):
    data=pd.read_csv(root/'datos_historicos/mensual.csv',parse_dates=['fecha'])
    rows=[];result={}
    for ind in IND:
        national=data[data.ciudad=='Nacional'].set_index('fecha')[ind].sort_index()
        for area in AREAS:
            orig=data[data.ciudad==area].set_index('fecha')[ind].sort_index()
            filled=orig.copy(); mask=orig.isna() & national.notna()
            # Sustitución explícita solicitada: NO transformar estimaciones en observados.
            filled.loc[mask]=national.loc[mask]
            for date in orig.index:
                rows.append(dict(ciudad=area,indicador=ind,fecha=str(date.date()),observado=orig.loc[date],
                       usado=filled.loc[date],origen='nacional_sustituto' if mask.loc[date] else ('observado' if pd.notna(orig.loc[date]) else 'no_disponible')))
            y=filled.loc[filled.first_valid_index():].asfreq('MS')
            assert y.notna().all() and (y>0).all() and y.index[-1]==pd.Timestamp('2025-12-01')
            result[(area,ind)]=y
    pd.DataFrame(rows).to_csv(out/'series_con_origen.csv',index=False)
    return data,result

def evaluate_one(y,area,ind,out,digest):
    key=f'{AREAS.index(area)}_{ind}';dest=out/key;dest.mkdir(exist_ok=True)
    if (dest/'completo.json').exists() and json.loads((dest/'completo.json').read_text(encoding='utf-8')).get('digest')==digest:
        print('Cache',area,ind,flush=True);return
    print('Iniciando',area,ind,flush=True);started=time.time()
    folds=[2022,2023,2024]; regrows=[]; validations=[]; pp=[];fails=[];diag=[];rs=[];information=[]
    # Exhaustiva dentro del conjunto predeclarado, todos los tamaños 1..8.
    for cols in SUBSETS:
        allerr=[];ok=True
        for year in folds:
            train=y.loc[:f'{year-1}-12'];target=y.loc[f'{year}-01':f'{year}-12']
            with np.errstate(over='ignore',invalid='ignore'):
                p,r,meta=regfit(train,cols)
            mt=metrics(target,p,train) if np.isfinite(p).all() else {k:float('inf') for k in ['RMSE','MAE','MAPE','MASE']}
            # Un predictor constante pre-2022 no es identificable en ese origen.
            identifiable=meta['rango']==meta['columnas']; ok &= identifiable
            regrows.append(dict(variables='+'.join(cols),k=len(cols),anio=year,identificable=identifiable,**mt))
            allerr.extend((target.values-p).tolist())
        validations.append(dict(modelo='REG:'+ '+'.join(cols),RMSE=float(np.sqrt(np.mean(np.square(allerr)))),elegible=bool(ok)))
    regdf=pd.DataFrame(regrows);regdf.to_csv(dest/'regresion_combinaciones.csv',index=False)
    validreg=sorted([v for v in validations if v['elegible'] and np.isfinite(v['RMSE'])],key=lambda v:(v['RMSE'],v['modelo']))
    bestcols=validreg[0]['modelo'][4:].split('+');valrows=[];by_model={}
    savejson(dest/'regresion_elegida.json',{'variables':bestcols,'RMSE_validacion':validreg[0]['RMSE'],
                 'combinaciones':255,'elegibles':len(validreg)})
    for name in MODELS:
        errors=[];foldgood=True
        for year in folds:
            tr=y.loc[:f'{year-1}-12'];te=y.loc[f'{year}-01':f'{year}-12']
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');p,r,meta=fitmodel(tr,name,cols=bestcols)
                for w in ws: fails.append(dict(modelo=name,etapa=str(year),tipo='aviso',mensaje=str(w.message)))
                if not np.isfinite(p).all() or (p<=0).any(): raise ValueError('Pronóstico no finito o no positivo')
                if meta.get('convergio',True)==False: foldgood=False
                errors.extend((te.values-p).tolist())
                valrows.append(dict(modelo=name,anio=year,**metrics(te,p,tr),convergio=meta.get('convergio',True)))
                pp.extend(dict(modelo=name,etapa='validacion',fecha=str(date.date()),observado=float(a),pred=float(b)) for date,a,b in zip(te.index,te.values,p))
            except Exception as e:
                foldgood=False;fails.append(dict(modelo=name,etapa=str(year),tipo='error',mensaje=str(e)))
        by_model[name]={'modelo':name,'RMSE_validacion':float(np.sqrt(np.mean(np.square(errors)))) if len(errors)==36 else float('inf'),
                        'elegible':bool(foldgood and len(errors)==36)}
        print('  validado',name,f"{by_model[name]['RMSE_validacion']:.0f}",flush=True)
        pd.DataFrame(fails,columns=['modelo','etapa','tipo','mensaje']).to_csv(dest/'avisos.csv',index=False)
    ranked=sorted([v for v in by_model.values() if v['elegible']],key=lambda v:(v['RMSE_validacion'],v['modelo']))
    winner=ranked[0]['modelo']
    # Se congela la decisión ANTES de consultar errores de 2025.
    savejson(dest/'decision_pre_prueba.json',dict(ganador=winner,ranking=ranked,variables=bestcols,cortes=folds))
    summaries=[];future=[];intervals=[];extra={}
    for name in MODELS:
        try:
            tr=y.loc[:'2024-12'];te=y.loc['2025-01':]; p,r,meta=fitmodel(tr,name,cols=bestcols)
            pp.extend(dict(modelo=name,etapa='prueba',fecha=str(date.date()),observado=float(a),pred=float(b)) for date,a,b in zip(te.index,te.values,p))
            summaries.append(dict(**by_model[name],ganador=name==winner,**metrics(te,p,tr),convergio_prueba=meta.get('convergio',True)))
            diag.append(dict(modelo=name,**diagnostics(r)))
            rs.extend(dict(modelo=name,paso=i,residuo=float(v)) for i,v in enumerate(r))
            information.append(dict(modelo=name,**{k:v for k,v in meta.items() if k in ['aic','aicc','bic','D','convergio']}))
            pf,rf,mf=fitmodel(y,name,cols=bestcols)
            extra[name]=mf
            future.extend(dict(modelo=name,fecha=str(date.date()),pred=float(v),ganador=name==winner) for date,v in zip(pd.date_range('2026-01-01',periods=12,freq='MS'),pf))
            if name==winner:
                # Banda orientativa: errores absolutos de origen móvil, por bloques de horizonte.
                vp=pd.DataFrame([row for row in pp if row['modelo']==winner and row['etapa']=='validacion'])
                vp['h']=pd.to_datetime(vp.fecha).dt.month
                for h,(date,v) in enumerate(zip(pd.date_range('2026-01-01',periods=12,freq='MS'),pf),1):
                    block=(h-1)//4;rows=vp[(vp.h-1)//4==block]
                    width=float(np.quantile(np.abs(np.log(rows.observado/rows.pred)),.90,method='higher'))
                    intervals.append(dict(fecha=str(date.date()),pred=float(v),inferior=float(v*np.exp(-width)),superior=float(v*np.exp(width)),n_calibracion=len(rows)))
        except Exception as e:
            fails.append(dict(modelo=name,etapa='prueba_final',tipo='error',mensaje=str(e)))
            if name==winner: raise
    for filename,rows in [('validacion',valrows),('comparacion',summaries),('predicciones',pp),('pronosticos_2026',future),
                          ('diagnosticos',diag),('residuos',rs),('criterios_informacion',information),('bandas_ganador',intervals)]:
        pd.DataFrame(rows).to_csv(dest/(filename+'.csv'),index=False)
    pd.DataFrame(fails,columns=['modelo','etapa','tipo','mensaje']).to_csv(dest/'avisos.csv',index=False)
    savejson(dest/'extras.json',extra)
    savejson(dest/'completo.json',dict(digest=digest,ciudad=area,indicador=ind,ganador=winner,segundos=time.time()-started))
    print('TERMINADO',area,ind,winner,round(time.time()-started),flush=True)

def reconstruction(data,series,out):
    recon=[];validation=[]
    # Relación ciudad/nacional estimada ANTES del evento; la nacional durante el evento es información retrospectiva.
    for ind in IND:
        nat=series[('Nacional',ind)]
        for area in AREAS[1:]:
            orig=data[data.ciudad==area].set_index('fecha')[ind]
            def bridge(end,dates):
                idx=orig.loc['2012-01-01':end].dropna().index
                X=np.column_stack([np.ones(len(idx)),np.log(nat.loc[idx]),np.sin(2*np.pi*idx.month/12),np.cos(2*np.pi*idx.month/12)])
                m=sm.OLS(np.log(orig.loc[idx]),X).fit(cov_type='HAC',cov_kwds={'maxlags':12})
                XF=np.column_stack([np.ones(len(dates)),np.log(nat.loc[dates]),np.sin(2*np.pi*dates.month/12),np.cos(2*np.pi*dates.month/12)])
                pr=m.get_prediction(XF).summary_frame()
                return np.exp(pr['mean'].values),np.exp(pr['obs_ci_lower'].values),np.exp(pr['obs_ci_upper'].values)
            for year in [2017,2018,2019]:
                dates=pd.date_range(f'{year}-03-01',periods=5,freq='MS');p,lo,hi=bridge(f'{year}-02-01',dates)
                for method,pred in [('Puente nacional',p),('Nacional directo',nat.loc[dates].values)]:
                    validation.append(dict(ciudad=area,indicador=ind,anio=year,metodo=method,**metrics(orig.loc[dates],pred,orig.loc[:f'{year}-02'].dropna())))
            dates=pd.date_range('2020-03-01',periods=5,freq='MS');p,lo,hi=bridge('2020-02-01',dates)
            for date,a,b,c in zip(dates,p,lo,hi):
                recon.append(dict(ciudad=area,indicador=ind,fecha=str(date.date()),reconstruido=a,inferior=b,superior=c,nacional=nat.loc[date]))
    pd.DataFrame(recon).to_csv(out/'reconstruccion_2020.csv',index=False)
    pd.DataFrame(validation).to_csv(out/'validacion_reconstruccion.csv',index=False)

def intervention(series,out):
    curves=[];coefs=[];diagn=[]
    for ind in IND:
        y=series[('Nacional',ind)].loc['2015-01':'2021-12'];idx=y.index
        t=np.arange(len(y));post=(idx>=pd.Timestamp('2020-03-01')).astype(float);t0=np.where(post)[0][0]
        X=pd.DataFrame({'const':1.,'t':t,'nivel':post,'pendiente':np.maximum(0,t-t0),
          'pulso':((idx>=pd.Timestamp('2020-03-01'))&(idx<=pd.Timestamp('2020-07-01'))).astype(float)},index=idx)
        for month in range(2,13): X[f'mes{month}']=(idx.month==month).astype(float)
        m=sm.OLS(np.log(y),X).fit(cov_type='HAC',cov_kwds={'maxlags':12})
        X0=X.copy();X0[['nivel','pendiente','pulso']]=0
        cf=np.exp(m.predict(X0));fit=np.exp(m.predict(X));ci=m.get_prediction(X0).conf_int()
        for key in ['nivel','pendiente','pulso']:
            c=m.conf_int().loc[key];coefs.append(dict(indicador=ind,termino=key,coef=m.params[key],p=m.pvalues[key],inferior=c.iloc[0],superior=c.iloc[1]))
        diagnostic=diagnostics(m.resid);diagn.append(dict(indicador=ind,modelo='ITS HAC',**diagnostic))
        # Contrafactual de segundo método: SARIMA ajustado SOLO hasta febrero 2020.
        prior=np.log(y.loc[:'2020-02']);postidx=idx[idx>=pd.Timestamp('2020-03')]
        mod=SARIMAX(prior,order=(0,1,1),seasonal_order=(1,0,0,12)).fit(disp=False,maxiter=150)
        fc=mod.get_forecast(len(postidx)); band=np.exp(fc.conf_int().values);pred=np.exp(fc.predicted_mean.values)
        cum=0.
        for i,date in enumerate(idx):
            j=i-t0;effect=float(fit.iloc[i]-cf.iloc[i]) if j>=0 else np.nan
            if j>=0:cum+=effect
            curves.append(dict(indicador=ind,fecha=str(date.date()),observado=y.iloc[i],ajuste=fit.iloc[i],contrafactual=cf.iloc[i],
               cf_inferior=np.exp(ci[i,0]),cf_superior=np.exp(ci[i,1]),efecto=effect,acumulado=cum if j>=0 else np.nan,
               precontrafactual=pred[j] if j>=0 else np.nan,pre_inferior=band[j,0] if j>=0 else np.nan,
               pre_superior=band[j,1] if j>=0 else np.nan,
               anomalia=bool(j>=0 and (y.iloc[i]<band[j,0] or y.iloc[i]>band[j,1]))))
    pd.DataFrame(curves).to_csv(out/'intervencion.csv',index=False)
    pd.DataFrame(coefs).to_csv(out/'intervencion_coeficientes.csv',index=False)
    pd.DataFrame(diagn).to_csv(out/'intervencion_diagnosticos.csv',index=False)

def exploration(series,out):
    records=[];summ=[];annual=[]
    for (area,ind),y in series.items():
        train=np.log(y.loc[:'2021-12'])
        for transform,z in [('log',train),('d1',train.diff().dropna()),('d1D12',train.diff().diff(12).dropna())]:
            with warnings.catch_warnings(record=True):
                a=adfuller(z,autolag='AIC');k=kpss(z,regression='c',nlags='auto');za=zivot_andrews(z,regression='ct',autolag='AIC')
            records.append(dict(ciudad=area,indicador=ind,transformacion=transform,ADF_p=a[1],KPSS_p=k[1],ZA_p=za[1],quiebre=str(z.index[int(za[4])].date())))
        summ.append(dict(ciudad=area,indicador=ind,**y.describe().to_dict()))
        for year,g in y.groupby(y.index.year): annual.append(dict(ciudad=area,indicador=ind,anio=year,meses=len(g),media_mensual=g.mean(),min=g.min(),max=g.max()))
    pd.DataFrame(records).to_csv(out/'estacionariedad.csv',index=False)
    pd.DataFrame(summ).to_csv(out/'descriptivas.csv',index=False)
    pd.DataFrame(annual).to_csv(out/'resumen_anual_series.csv',index=False)

def run(root):
    root=Path(root);out=root/'resultados_series';out.mkdir(exist_ok=True)
    digest=hashlib.sha256(Path(__file__).read_bytes()+(root/'datos_historicos/mensual.csv').read_bytes()).hexdigest()
    if (out/'resumen.json').exists() and json.loads((out/'resumen.json').read_text(encoding='utf-8')).get('digest')==digest:
        print('Resultados temporales vigentes',flush=True);return
    data,series=load_series(root,out)
    workers=int(os.environ.get('GEIH_SERIES_WORKERS','2'))
    if workers==1:
        for (area,ind),y in series.items(): evaluate_one(y,area,ind,out,digest)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(evaluate_one,y,area,ind,out,digest) for (area,ind),y in series.items()]
            for f in futures:f.result()
    reconstruction(data,series,out);intervention(series,out);exploration(series,out)
    allcomp=[];allfuture=[]
    for area in AREAS:
        for ind in IND:
            dest=out/f'{AREAS.index(area)}_{ind}'
            c=pd.read_csv(dest/'comparacion.csv');c.insert(0,'indicador',ind);c.insert(0,'ciudad',area);allcomp.append(c)
            f=pd.read_csv(dest/'pronosticos_2026.csv');f.insert(0,'indicador',ind);f.insert(0,'ciudad',area);allfuture.append(f)
    pd.concat(allcomp).to_csv(out/'comparacion_todas_series.csv',index=False)
    forecast=pd.concat(allfuture);forecast.to_csv(out/'pronosticos_todas_series.csv',index=False)
    # Auditoría de orden; las predicciones marginales independientes pueden cruzarse.
    w=forecast[forecast.ganador].pivot(index=['ciudad','fecha'],columns='indicador',values='pred')
    w['cruce']=(w.p10>w.mediana)|(w.mediana>w.p90);w.reset_index().to_csv(out/'auditoria_cruces.csv',index=False)
    savejson(out/'resumen.json',dict(digest=digest,series=24,modelos=len(MODELS),regresiones_por_serie=255,
             validacion=[2022,2023,2024],prueba=2025,pronostico=2026,cruces=int(w.cruce.sum()),inicio='2008-07',fin='2025-12'))

if __name__=='__main__': run(sys.argv[1] if len(sys.argv)>1 else Path.cwd())
