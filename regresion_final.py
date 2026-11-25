"""Regresión por dominio: 511 subconjuntos, Ridge y Lasso, selección por validación."""
from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from threadpoolctl import threadpool_limits
from modelado_historico import particionar
from ml_final import AREAS
threadpool_limits(2)
NUM=['edad','antiguedad','horas','anio'];CAT=['sexo','jefe','estudia','estado_civil','educacion'];VARS=NUM+CAT

def run(root):
    root=Path(root)
    for area in AREAS:
        out=root/'resultados_regresion_final'/str(AREAS.index(area));out.mkdir(parents=True,exist_ok=True)
        source=root/('datos_ml_nacional/base_individual.parquet' if area=='Nacional' else 'resultados_historicos/base_individual.parquet')
        digest=hashlib.sha256(Path(__file__).read_bytes()+source.read_bytes()).hexdigest()
        if (out/'resumen.json').exists() and json.loads((out/'resumen.json').read_text(encoding='utf-8')).get('digest')==digest:continue
        print('Regresión final',area,flush=True);d=pd.read_parquet(source)
        if area!='Nacional':
            d=d[d.ciudad==area].merge(pd.read_parquet(root/'resultados_historicos/particion.parquet')[['id_persona','conjunto']],on='id_persona',validate='one_to_one')
        else:d=particionar(d,out)
        tr=d.conjunto.eq('entrenamiento').values;va=d.conjunto.eq('validacion').values;te=d.conjunto.eq('prueba').values
        blocks=[];groups={};names=['const'];offset=1;processors={}
        for v in VARS:
            arr=d[[v]].astype(float) if v in NUM else d[[v]].astype(object).where(d[[v]].notna(),np.nan)
            im=SimpleImputer(strategy='median' if v in NUM else 'most_frequent').fit(arr.loc[tr]);clean=im.transform(arr)
            proc=StandardScaler() if v in NUM else OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)
            proc.fit(clean[tr]);b=proc.transform(clean);blocks.append(b);processors[v]=(im,proc)
            cols=[v] if v in NUM else list(proc.get_feature_names_out([v]));names.extend(cols);groups[v]=list(range(offset,offset+len(cols)));offset+=len(cols)
        X=np.column_stack([np.ones(len(d)),*blocks]);del blocks
        y=d.ingreso.to_numpy();A=X[tr];Y=y[tr];B=X[va];V=y[va];ATA=A.T@A;ATY=A.T@Y;BTB=B.T@B;BTV=B.T@V
        yy=Y@Y;vv=V@V;sst=((Y-Y.mean())**2).sum();vst=((V-V.mean())**2).sum();rows=[];coeff=[]
        for k in range(1,10):
            for subset in itertools.combinations(VARS,k):
                ix=[0]+[j for v in subset for j in groups[v]];b=np.linalg.lstsq(ATA[np.ix_(ix,ix)],ATY[ix],rcond=None)[0]
                rss=max(0.,yy-2*b@ATY[ix]+b@ATA[np.ix_(ix,ix)]@b);vss=max(0.,vv-2*b@BTV[ix]+b@BTB[np.ix_(ix,ix)]@b)
                rows.append(dict(k=k,variables='+'.join(subset),RMSE=np.sqrt(vss/len(V)),R2_validacion=1-vss/vst,R2_ajustado=1-rss/sst*(len(Y)-1)/(len(Y)-len(ix)),AIC=len(Y)*np.log(rss/len(Y))+2*len(ix),BIC=len(Y)*np.log(rss/len(Y))+np.log(len(Y))*len(ix),SCT=sst,SCR=sst-rss,SCE=rss))
                coeff.append((ix,b))
        grid=pd.DataFrame(rows);grid.to_csv(out/'combinaciones.csv',index=False);winner=int(grid.RMSE.idxmin());ix,b=coeff[winner]
        best_sizes=grid.sort_values('RMSE').groupby('k').head(1).sort_values('k').copy()
        best_sizes['MAE']=[mean_absolute_error(V,B[:,coeff[j][0]]@coeff[j][1]) for j in best_sizes.index];best_sizes.to_csv(out/'mejores_por_tamano.csv',index=False)
        reg=[];models={}
        for name,cls,alphas in [('Ridge',Ridge,[.1,1,10,100]),('Lasso',Lasso,[10,100,1000,10000])]:
            for alpha in alphas:
                kw={'max_iter':5000} if name=='Lasso' else {}
                m=cls(alpha=alpha,**kw).fit(A[:,1:],Y);pr=m.predict(B[:,1:]);reg.append(dict(modelo=name,alpha=alpha,RMSE=np.sqrt(mean_squared_error(V,pr)),MAE=mean_absolute_error(V,pr),R2=r2_score(V,pr)))
                key=f'{name}_{alpha}';models[key]=m
        table=pd.DataFrame(reg);table.to_csv(out/'regularizacion.csv',index=False)
        candidates=[dict(modelo='OLS subconjunto',alpha=np.nan,RMSE=grid.loc[winner,'RMSE'])]
        for name in ['Ridge','Lasso']:candidates.append(table[table.modelo==name].sort_values('RMSE').iloc[0].to_dict())
        choice=min(candidates,key=lambda a:a['RMSE'])
        (out/'decision_pre_prueba.json').write_text(json.dumps(dict(variables=grid.loc[winner,'variables'],ganador=choice),default=str,ensure_ascii=False,indent=2),encoding='utf-8')
        test=[];predictions=[]
        for row in candidates:
            name=row['modelo'];pred=X[te][:,ix]@b if name=='OLS subconjunto' else models[f"{name}_{row['alpha']:g}"].predict(X[te][:,1:])
            test.append(dict(modelo=name,ganador=name==choice['modelo'],RMSE=np.sqrt(mean_squared_error(y[te],pred)),MAE=mean_absolute_error(y[te],pred),R2=r2_score(y[te],pred),negativos=int((pred<0).sum())))
            if name=='OLS subconjunto':
                sample=pd.DataFrame(dict(observado=y[te],pred=pred)).sample(min(5000,te.sum()),random_state=78018);sample.to_csv(out/'grafico.csv',index=False)
        pd.DataFrame(test).to_csv(out/'prueba.csv',index=False)
        # HC1 por lotes: evitar una matriz n x p adicional para toda la muestra.
        meat=np.zeros((len(ix),len(ix)));sse=0
        for start in range(0,len(Y),10000):
            xx=A[start:start+10000][:,ix];r=Y[start:start+10000]-xx@b;meat+=(xx*r[:,None]).T@(xx*r[:,None]);sse+=r@r
        inv=np.linalg.pinv(ATA[np.ix_(ix,ix)]);se=np.sqrt(np.maximum(0,np.diag(inv@meat@inv)*len(Y)/(len(Y)-len(ix))))
        from scipy.stats import norm
        pd.DataFrame(dict(variable=np.array(names)[ix],coef=b,se=se,p=2*norm.sf(np.abs(b/se)),lo95=b-1.96*se,hi95=b+1.96*se)).to_csv(out/'coeficientes_OLS.csv',index=False)
        joblib.dump(dict(preprocesamiento=processors,variables=VARS,columnas_OLS=ix,coef_OLS=b,modelos_regularizados=models,decision=choice),out/'modelos.joblib')
        (out/'resumen.json').write_text(json.dumps(dict(digest=digest,ciudad=area,n=len(d),combinaciones=len(grid),variables_OLS=grid.loc[winner,'variables'],ganador=choice['modelo']),ensure_ascii=False,indent=2),encoding='utf-8')
        print('REG TERMINADA',area,choice['modelo'],flush=True)

if __name__=='__main__':run(sys.argv[1])
