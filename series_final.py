"""Selección sin ARCH/GARCH, intervalos 80/95 y contrafactuales de seis dominios."""
from pathlib import Path
import json,sys,hashlib,warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
from series_temporales import AREAS,IND,SAR,SEED,fitmodel,regfit,features,savejson

def exog(index):
    return np.column_stack([(index>=pd.Timestamp('2022-01-01')).astype(float),
          ((index>=pd.Timestamp('2020-03-01'))&(index<=pd.Timestamp('2020-07-01'))).astype(float)])

def blocks(residual,n=2000,h=12):
    r=np.asarray(residual);r=r[np.isfinite(r)];r=r-r.mean();rng=np.random.default_rng(SEED)
    starts=rng.integers(0,len(r),size=n)
    return r[(starts[:,None]+np.arange(h))%len(r)]

def lstm_paths(y,epochs,n=2000):
    import torch
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    log=np.log(y.values);dz=np.diff(log);mu=dz.mean();sd=max(dz.std(),1e-6);u=(dz-mu)/sd;w=12
    X=torch.tensor(np.array([u[i-w:i] for i in range(w,len(u))]),dtype=torch.float32).unsqueeze(-1)
    target=torch.tensor(u[w:],dtype=torch.float32).unsqueeze(-1)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.rnn=torch.nn.LSTM(1,16,batch_first=True);self.out=torch.nn.Linear(16,1)
        def forward(self,x):return self.out(self.rnn(x)[0][:,-1,:])
    model=Net();opt=torch.optim.Adam(model.parameters(),lr=.01)
    for _ in range(epochs):
        opt.zero_grad();loss=((model(X)-target)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    with torch.no_grad():
        r=(target-model(X)).numpy().ravel()*sd;innov=blocks(r,n)
        hist=np.tile(u[-12:],(n,1));level=np.repeat(log[-1],n);paths=[]
        for h in range(12):
            v=model(torch.tensor(hist[:,-12:],dtype=torch.float32).unsqueeze(-1)).numpy().ravel()
            change=v*sd+mu+innov[:,h];level=level+change;paths.append(level.copy());hist=np.column_stack([hist,(change-mu)/sd])
    return np.exp(np.asarray(paths)),model.state_dict()

def intervals(y,name,cols):
    fut=pd.date_range(y.index[-1]+pd.offsets.MonthBegin(),periods=12,freq='MS');pred,resid,meta=fitmodel(y,name,cols=cols)
    actualname=meta.get('seleccion_AICc',name);parameters=[];weightstate=None
    if actualname in SAR:
        order,season=SAR[actualname];isx=actualname.startswith('SARIMAX')
        m=SARIMAX(np.log(y),order=order,seasonal_order=season,exog=exog(y.index) if isx else None,
                 enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=150)
        f=m.get_forecast(12,exog=exog(fut) if isx else None);a=np.exp(f.conf_int(alpha=.2).values);b=np.exp(f.conf_int(alpha=.05).values)
        for k in m.params.index:parameters.append(dict(parametro=k,estimacion=m.params[k],se=m.bse[k],p=m.pvalues[k]))
        bounds=np.column_stack([a[:,0],a[:,1],b[:,0],b[:,1]]);method='SARIMA: distribución predictiva gaussiana en log, parámetros fijos'
    else:
        if name=='Regresion exhaustiva':
            _,resid,meta=regfit(y,cols);coef=meta['coeficientes'];z=np.log(y.values);n=2000;hist=np.tile(z,(n,1));innov=blocks(resid,n)
            for h,date in enumerate(fut):
                # Variables de calendario compartidas y rezagos recursivos por trayectoria.
                row=features(hist[0],date);value=np.repeat(coef['intercepto'],n)
                for v in cols:value+=coef[v]*(hist[:,-int(v[3:])] if v.startswith('lag') else row[v])
                hist=np.column_stack([hist,value+innov[:,h]])
            paths=np.exp(hist[:,-12:].T);method='Bootstrap circular de innovaciones, bloque 12, recursivo, coeficientes fijos'
            X=np.array([[1]+[features(z[:i],y.index[i])[v] for v in cols] for i in range(12,len(z))]);m=sm.OLS(z[12:],X).fit(cov_type='HAC',cov_kwds={'maxlags':12})
            for j,k in enumerate(['intercepto',*cols]):parameters.append(dict(parametro=k,estimacion=m.params[j],se=m.bse[j],p=m.pvalues[j]))
        elif name.startswith('LSTM'):
            paths,weightstate=lstm_paths(y,int(name.split()[1]));method='LSTM: bootstrap circular recursivo de innovaciones, pesos fijos'
        elif name.startswith('Prophet'):
            from prophet import Prophet
            np.random.seed(SEED);m=Prophet(changepoint_prior_scale=float(name.split()[1]),yearly_seasonality=4,weekly_seasonality=False,daily_seasonality=False,uncertainty_samples=2000)
            m.fit(pd.DataFrame({'ds':y.index,'y':np.log(y.values)}),seed=SEED)
            paths=np.exp(m.predictive_samples(pd.DataFrame({'ds':fut}))['yhat']);method='Prophet MAP: simulación de tendencia futura y ruido, sin posterior de estacionalidad'
            for k,v in m.params.items():
                for j,value in enumerate(np.asarray(v).ravel()):parameters.append(dict(parametro=f'{k}[{j}]',estimacion=value,se=np.nan,p=np.nan))
        elif name in ['SES','Holt','Holt amortiguado','ETS AAA','ETS AAdA']:
            m=ExponentialSmoothing(np.log(y),trend=None if name=='SES' else 'add',damped_trend=name in ['Holt amortiguado','ETS AAdA'],seasonal='add' if name.startswith('ETS') else None,seasonal_periods=12 if name.startswith('ETS') else None,initialization_method='estimated').fit()
            paths=np.exp(np.asarray(m.simulate(12,anchor='end',repetitions=2000,random_errors=blocks(m.resid).T)));method='ETS: simulación de estados con innovaciones bootstrap, parámetros fijos'
        else:raise ValueError('Ganador no soportado: '+name)
        bounds=np.quantile(paths,[.1,.9,.025,.975],axis=1).T
    frame=pd.DataFrame(dict(fecha=fut,pred=pred,lo80=bounds[:,0],hi80=bounds[:,1],lo95=bounds[:,2],hi95=bounds[:,3],metodo=method))
    return frame,parameters

def run(root):
    root=Path(root);old=root/'resultados_series';out=root/'resultados_series_final';out.mkdir(exist_ok=True)
    digest=hashlib.sha256(Path(__file__).read_bytes()+(old/'resumen.json').read_bytes()).hexdigest()
    if (out/'resumen.json').exists() and json.loads((out/'resumen.json').read_text(encoding='utf-8')).get('digest')==digest:return
    d=pd.read_csv(old/'series_con_origen.csv',parse_dates=['fecha']);comp=pd.read_csv(old/'comparacion_todas_series.csv')
    comp['seleccionable']=comp.elegible&~comp.modelo.str.startswith('AR1-');chosen=comp[comp.seleccionable].sort_values(['RMSE_validacion','modelo']).groupby(['ciudad','indicador']).head(1)
    comp['ganador_final']=False
    for i in chosen.index:comp.loc[i,'ganador_final']=True
    comp.to_csv(out/'comparacion.csv',index=False);chosen.to_csv(out/'seleccion_pre_prueba.csv',index=False)
    forecasts=[];parameters=[];coverage=[];counter=[];eda=[]
    for area in AREAS:
        for ind in IND:
            print('SERIE FINAL',area,ind,flush=True)
            p=old/f'{AREAS.index(area)}_{ind}';decision=json.loads((p/'decision_pre_prueba.json').read_text(encoding='utf-8'))
            row=chosen[(chosen.ciudad==area)&(chosen.indicador==ind)].iloc[0];g=d[(d.ciudad==area)&(d.indicador==ind)].set_index('fecha')
            y=g.usado.dropna().asfreq('MS')
            for stage,yy in [('prueba',y.loc[:'2024-12']),('pronostico',y)]:
                f,params=intervals(yy,row.modelo,decision['variables']);f['ciudad']=area;f['indicador']=ind;f['modelo']=row.modelo;f['etapa']=stage;forecasts.append(f)
                parameters.extend(dict(ciudad=area,indicador=ind,modelo=row.modelo,etapa=stage,**v) for v in params)
                if stage=='prueba':
                    actual=y.loc['2025'].values
                    coverage.append(dict(ciudad=area,indicador=ind,cobertura80=np.mean((actual>=f.lo80)&(actual<=f.hi80)),cobertura95=np.mean((actual>=f.lo95)&(actual<=f.hi95)),ancho80=np.mean(f.hi80-f.lo80),ancho95=np.mean(f.hi95-f.lo95)))
            returns=100*np.log(y.loc[:'2021-12']).diff().dropna();r=returns-returns.mean()
            eda.append(dict(ciudad=area,indicador=ind,LB_cuadrados=float(acorr_ljungbox(r**2,lags=[12],return_df=True).lb_pvalue.iloc[0]),ARCH_p=float(het_arch(r,nlags=12)[1])))
            # Escenario urbano sin pandemia: se aprende únicamente antes del choque, sin imputados de 2020.
            prior=np.log(y.loc['2015-01':'2020-02']);m=SARIMAX(prior,order=(0,1,1),seasonal_order=(1,0,0,12)).fit(disp=False,maxiter=150)
            fc=m.get_forecast(22);dates=fc.predicted_mean.index;cf=np.exp(fc.predicted_mean.values);c80=np.exp(fc.conf_int(alpha=.2).values);c95=np.exp(fc.conf_int(alpha=.05).values)
            observed=g.reindex(dates).observado.to_numpy();used=g.reindex(dates).usado.to_numpy()
            effect=observed-cf;cum=pd.Series(effect).cumsum().to_numpy()
            for j,date in enumerate(dates):counter.append(dict(ciudad=area,indicador=ind,fecha=date,observado=observed[j],usado=used[j],contrafactual=cf[j],lo80=c80[j,0],hi80=c80[j,1],lo95=c95[j,0],hi95=c95[j,1],desviacion_observada=effect[j],acumulado_observado=cum[j],sustituido=pd.isna(observed[j])))
            pd.concat(forecasts).to_csv(out/'intervalos.csv',index=False)
    pd.DataFrame(parameters).to_csv(out/'parametros.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'cobertura_intervalos.csv',index=False)
    pd.DataFrame(eda).to_csv(out/'diagnostico_volatilidad.csv',index=False)
    pd.DataFrame(counter).to_csv(out/'contrafactuales.csv',index=False)
    savejson(out/'resumen.json',dict(digest=digest,series=24,excluidos=['AR1-ARCH1','AR1-GARCH11'],h=12,origen='2025-12',pronostico=2026,simulaciones=2000))

if __name__=='__main__':run(sys.argv[1])
