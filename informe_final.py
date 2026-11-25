"""Informe integrado: todas las tablas proceden de resultados reproducibles.

Las funciones solo presentan resultados; no reescogen modelos con el test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from ml_final import AREAS, GRIDS, FINAL, CIUDADES, RESTO, ETIQUETA, FEATURES_FINAL, ficha_filas
import decision_costos as dc
from series_temporales import AREAS as TAREAS, IND, MODELS, SAR

LABEL={'p10':'P10','mediana':'P50 / mediana','p90':'P90','promedio':'Promedio'}
MODEL_TEXT={
'Logistica': 'Regresión logística multinomial: modela probabilidades de las cuatro clases mediante relaciones lineales en el logaritmo de sus razones. Se comparan C=0,1 y 1; penalización L2, solver lbfgs y máximo 1500 iteraciones. Los coeficientes son asociaciones condicionadas, no efectos causales. La relación lineal puede perder interacciones de educación y trayectoria laboral.',
'Arbol':'Árbol de decisión: particiona los perfiles mediante reglas. Se comparan profundidades 6 y 12; criterio Gini y mínimo 30 observaciones por hoja. Facilita describir reglas de segmentación, aunque un cambio de muestra puede cambiar sus cortes. Las hojas no representan contratos salariales ni derechos adquiridos.',
'Random Forest':'Bosque aleatorio: 100 árboles, profundidad máxima 14, mínimo 30 observaciones por hoja; max_features toma sqrt o 0,5. Promedia árboles con bootstrap y semilla fija. Admite interacciones no lineales y reduce la variabilidad de un árbol; no convierte las asociaciones en causas.',
'SVM lineal':'SVM lineal: margen uno contra el resto, C=0,1 o 1, dual=False y máximo 5000 iteraciones. Usa variables escaladas. Sus márgenes permiten ordenar casos y calcular ROC/PR, pero no son probabilidades calibradas. Aquí se utiliza el SVM lineal de la actualización previa; no se presenta como un SVM RBF.',
'ACP + SVM':'ACP seguido de SVM lineal: retención de 80 % o 95 % de la varianza, descomposición completa y C=1. El ACP se aprende exclusivamente dentro de cada entrenamiento, seguido del clasificador. Reducir varianza redundante puede simplificar el espacio, pero una componente de poca varianza puede ser predictiva. ACP por sí solo no clasifica ingresos.',
'KNN':'Vecinos cercanos: k=5,15,31,51,75; voto uniforme, distancia euclídea, kd_tree y leaf_size=60. Por costo computacional el ajuste usa hasta 50 000 filas estratificadas del entrenamiento, con semilla fija; validación y prueba se evalúan completas. La comparación refleja este presupuesto y no un KNN entrenado con todas las filas.',
'Red MLP':'Red neuronal multicapa: configuraciones (32 neuronas, alpha=0,001) y (32,16 neuronas, alpha=0,01), activación ReLU, salida multiclase, Adam, tasa inicial 0,001, lote 1024, máximo 60 épocas. Dentro de cada entrenamiento se reserva 10 % por grupos para parada temprana; paciencia 6 y mejora mínima de log-loss 0,0001. Se restauran los mejores pesos; no se reutilizan validación externa ni prueba para detenerla. No se calculan p-valores clásicos para sus pesos.'}

def loadj(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def table(df,caption=None):
    """Nunca redondear años, meses, pliegues o conteos a cifras significativas."""
    x=df.copy()
    for c in x.select_dtypes(include=['bool']):x[c]=x[c].map({True:'Sí',False:'No'})
    for c in x.select_dtypes(include=['object','string']):
        x[c]=x[c].map(lambda v:v.replace('+',', ') if isinstance(v,str) and c=='variables' else v)
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_bool_dtype(x[c]):
            vals=x[c].dropna()
            integer=pd.api.types.is_integer_dtype(x[c])
            money=any(w in c.lower() for w in ['rmse','mae','ingreso','pred','lo80','hi80','lo95','hi95','media_mensual','ancho']) and not any(w in c.lower() for w in ['clase','predicha'])
            def fmt(v):
                if pd.isna(v):return '—'
                if not np.isfinite(v):return 'No elegible'
                if integer:return str(int(v))
                if money:return f'{v:,.0f}'.replace(',',' ')
                if 0<abs(v)<.001:return f'{v:.2e}'
                return f'{v:.3f}'
            x[c]=x[c].map(fmt)
    x=x.rename(columns={'modelo':'Modelo','ciudad':'Dominio','indicador':'Indicador','pred':'Puntual','lo80':'LI80','hi80':'LS80','lo95':'LI95','hi95':'LS95','f1_macro':'F1 macro','balanceada':'Exact. balanceada','exactitud':'Exactitud','precision_macro':'Precisión macro','auc_macro':'AUC macro','RMSE_validacion':'RMSE validación','ganador_final':'Seleccionado'})
    s=x.to_markdown(index=False)
    return '\n\n'+s+'\n\n'+('Tabla: '+caption+'\n\n' if caption else '')

class Report:
    def __init__(self,root):
        self.root=Path(root);self.fig=self.root/'informe_final_figuras';self.fig.mkdir(exist_ok=True)
        self.parts=[];self.nfig=0
        plt.rcParams.update({'font.size':9,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    def add(self,s):self.parts.append(s+'\n\n')
    def tab(self,d,cap=None):self.add(table(d,cap))
    def picture(self,fig,cap,height=None):
        self.nfig+=1;p=self.fig/f'fig_{self.nfig:03d}.png';fig.tight_layout();fig.savefig(p,bbox_inches='tight');plt.close(fig)
        self.add(f'![{cap}]({p.relative_to(self.root).as_posix()})'+('{width=95%}' if height is None else '{height='+str(height)+'in}'))
    def csv(self,rel,**kw):return pd.read_csv(self.root/rel,**kw)
    def text(self):return '\n'.join(self.parts)

def build(root):
    R=Report(root);root=R.root
    stats=[loadj(root/'resultados_ml_final'/str(i)/'resumen.json') for i in range(6)]
    monthly=R.csv('datos_historicos/mensual.csv',parse_dates=['fecha'])
    R.add('# Introducción, justificación y preguntas\n\nEl ingreso laboral mensual permite estudiar diferencias de remuneración y capacidad de compra nominal de los trabajadores. La GEIH del DANE reúne información sobre empleo e ingresos; sus cambios de marco y los ajustes de recolección durante la pandemia hacen indispensable distinguir comportamiento económico de cambios en la medición [@dane2007historico; @danerediseno; @danecovid2020]. Este proyecto integra clasificación de personas y pronóstico de indicadores agregados, que responden preguntas distintas.\n\nLa primera pregunta, aceptada en la revisión del proyecto, es: **¿qué tan bien clasifica un único modelo nacional, con la ciudad como variable, el ingreso laboral de un asalariado en cuatro tramos (C1–C4) definidos por los cortes P10, P50 y P90 calculados en entrenamiento por ciudad y año?** Ese modelo final es el único que se despliega y el que describe la ficha; los modelos ajustados por separado en las cinco áreas se conservan como comparación. La clase se asigna con una regla de decisión basada en una matriz de costos que explicita qué error es más caro. La segunda es: **¿qué procedimiento pronostica mejor P10, P50, P90 y promedio mensual a doce meses, con origen diciembre de 2025?** Una tercera pregunta explora cuánto se apartaron las series de la trayectoria que un modelo previo a marzo de 2020 habría proyectado sin introducir el choque.\n\nEl horizonte es enero–diciembre de **2026**, el año siguiente al cierre de los datos. Aunque el informe se compila en noviembre de 2026, esta es una evaluación y proyección con información congelada hasta diciembre de 2025; no utiliza observaciones de 2026. La utilidad es orientar planificación de remuneraciones, presupuestos y diagnóstico de segmentación. Las predicciones requieren verificación antes de cualquier uso individual.')
    R.add('## Alcance y marco teórico\n\nP10, P50 y P90 son puntos de la distribución, no tres grupos exhaustivos. Se construyen cuatro clases: C1: ingreso ≤ P10; C2: P10 < ingreso ≤ P50; C3: P50 < ingreso ≤ P90; C4: ingreso > P90. Los empates permanecen juntos, por lo que sus frecuencias no tienen que ser 10/40/40/10 %. Los cortes de clasificación son **por ciudad y año y se calculan solo con entrenamiento**; los indicadores de series de tiempo son **mensuales**. En el modelo final la ciudad es un predictor con seis categorías (las cinco áreas y Resto del país); dentro de cada modelo de comparación por área es constante y no entra.\n\nLa clasificación supervisada aprende asociaciones entre perfiles y tramos; ACP sintetiza variabilidad de los predictores y K-means agrupa perfiles sin utilizar ingreso ni etiquetas. Una agrupación no equivale a una clase salarial [@sklearnsvm; @sklearnpca; @sklearncluster]. La regresión individual estima ingreso en COP. Los modelos temporales aprovechan persistencia, tendencia y estacionalidad; las redes neuronales permiten relaciones no lineales, pero requieren regularización y evaluación fuera de muestra. La intervención temporal y el contrafactual son análisis de escenarios, sin identificación causal automática [@statsmodelsseries; @prophetdiagnostico; @pytorchlstm].')
    R.add('## Antecedentes y relación con el estudio\n\nAbadía estudia diferencias ligadas al tipo contractual en Colombia: motiva documentar filtros de contrato, pero su diseño no equivale a clasificar cuantiles [@abadia2014]. Ki y Shang comparan predicción de salarios mínimos entre países con bosques y redes: apoyan contrastar familias no lineales, aunque salario mínimo nacional e ingreso individual son objetivos diferentes [@kishang2024]. Vera-Jaramillo propone reconstruir indicadores laborales subnacionales y distinguir registros observados de modelados; se retoma esa trazabilidad, sin trasladar sus tasas laborales al salario [@verajaramillo2025].\n\nBreiman fundamenta la agregación de árboles y la importancia de su fuerza y correlación; no garantiza que el bosque venza en esta muestra [@breiman2001]. REFORMS orienta el reporte explícito de datos, selección y evaluación para evitar afirmaciones que el diseño no sostiene [@kapoor2024]. Chernozhukov y colaboradores distinguen herramientas predictivas e identificación causal; por eso el escenario pandémico aquí no se presenta como efecto identificado [@chernozhukov2024]. Estas seis referencias cubren antecedentes económicos y metodológicos; no se comparan sus errores numéricos con los de GEIH porque sus poblaciones, respuestas y particiones difieren.')
    R.add('# Datos, ficha de las series y reproducibilidad de la muestra\n\nLa fuente es el Archivo Nacional de Datos del DANE. Se trabaja con microdatos armonizados y resúmenes derivados, sin ponderar, en COP corrientes. **No son estadísticas poblacionales oficiales del DANE.** La cobertura selecciona adultos asalariados públicos o privados con contrato escrito y duración contractual identificada; exige ingreso mensual válido ≥ 100 000 COP, horas semanales en (0,100] y antigüedad laboral plausible. Los códigos de no respuesta se transforman en faltantes. Los filtros definen la población analítica y pueden excluir trabajadores precarios: no representan a todos los ocupados.\n\nLas cinco áreas son Medellín AM, Barranquilla AM, Bogotá, Bucaramanga AM y Cali AM. La base nacional procede de la cobertura nacional disponible de los módulos oficiales; **no es la suma de las cinco áreas**. Esas bases se solapan y sus registros no se deben sumar como personas independientes. Cada registro es una persona-mes, no una persona seguida longitudinalmente.\n\nSe comprobó que los 496 357 registros de las cinco áreas están contenidos, con valores idénticos, en la base nacional, y que sus hogares no tienen otros miembros en ella. Para el modelo final, la variable ciudad toma el área de esos hogares y el valor **Resto del país** para los demás registros nacionales: la base nacional armonizada no conserva el código de las demás áreas metropolitanas, por lo que esa categoría agrupa otras ciudades y zonas.')
    R.tab(pd.DataFrame([{'Dominio':ETIQUETA[s['area']],'Registros':s['n'],'Entrenamiento':s['entrenamiento'],'Validación':s['validacion'],'Prueba':s['prueba']} for s in stats]),'Conteos efectivos después de filtros y partición por hogar-mes. El modelo final usa toda la base nacional.')
    counts=monthly.groupby(['anio','ciudad']).n.sum(min_count=1).unstack().reindex(columns=AREAS).fillna(0).astype(int).reset_index();R.tab(counts,'Registros seleccionados por año y dominio. Nacional se solapa con las áreas; 2008 es parcial y 2020 tiene cambios de cobertura.')
    R.tab(pd.DataFrame([
        ['edad','P6040','Años cumplidos; numérica'],
        ['antiguedad','P6426','Meses en el empleo; 998/999 se tratan como faltantes'],
        ['horas','P6800','Horas habituales por semana; numérica'],
        ['sexo','P6020 / P3271','Hombre/Mujer según codificación disponible'],
        ['jefe','P6050','Sí si es jefe del hogar; No para otros parentescos válidos'],
        ['estudia','P6170','Asistencia educativa: Sí/No'],
        ['estado_civil','P6070','Categorías 1–6 conservadas como nominales'],
        ['educacion','P6210 / P3042','Primaria o menos, Secundaria, Media, Superior'],
        ['ciudad','AREA (vía hogar)','Cinco áreas del estudio o Resto del país; predictor solo del modelo final'],
        ['ingreso','INGLABO','COP mensuales; respuesta, nunca predictor de su propia clase']
    ],columns=['Variable analítica','Fuente','Unidad o recodificación']),'Diccionario de entradas y respuesta. Las tablas de ejemplos y perfiles expresan antigüedad en meses, edad en años y horas por semana.')
    R.tab(pd.DataFrame([
        ['Nombre','P10, mediana/P50, P90 y promedio de INGLABO en la muestra seleccionada'],
        ['Código en la fuente','Variable INGLABO; no hay un ID DANE de estas 24 series derivadas'],
        ['Catálogo','GEIH 2007: catálogo 317; GEIH 2025: catálogo 853, ANDA'],
        ['Enlace','https://microdatos.dane.gov.co/index.php/catalog/853'],
        ['Fecha de descarga','Fecha original no registrada de forma uniforme; inventario y hashes incluidos'],
        ['Fecha de consulta del informe','22 de noviembre de 2026'],
        ['Unidades y frecuencia','COP corrientes por mes; frecuencia mensual'],
        ['Periodo solicitado','Enero de 2007 a diciembre de 2025: 228 meses de calendario'],
        ['Periodo utilizable','Julio de 2008 a diciembre de 2025: 210 meses'],
        ['Observaciones genuinas','Nacional: 210; cada área: 205; cinco faltantes urbanos en marzo–julio de 2020'],
        ['Ajuste estacional y base','Sin ajuste estacional previo; sin año base ni deflactación IPC']
    ],columns=['Campo','Descripción']))
    R.add('**Restricción de extensión:** incluso 2007–2025 completo tiene 228 meses, menos de los 240 exigidos. En la base auditada el ingreso usable empieza en julio de 2008; los meses anteriores permanecen faltantes. No se afirma que no existan ingresos recuperables en otras versiones: una auditoría adicional de variables históricas podría ampliar la base, pero este trabajo no los recuperó. La GEIH comenzó en 2006 y la transición con ECH no justifica inventar una serie homogénea anterior. El reporte necesita **excepción del docente, todavía no acreditada**. Se reserva un año completo para prueba y tres años para validación, pero esto no sustituye la exigencia de longitud.\n\nLa fecha original de descarga es una limitación de trazabilidad: se entrega el inventario de URLs y hashes disponible, junto con los datos analíticos congelados. Los CSV analíticos son productos de armonización, no se etiquetan como ZIP/CSV originales del DANE. La descarga oficial y los archivos originales se describen en las instrucciones de entrega.')
    R.add('## Faltantes, rupturas y transformaciones\n\nSe conserva una columna de observado, otra de usado y una marca de origen. Solo los cinco meses urbanos faltantes de marzo–julio de 2020 se sustituyen por el **mismo indicador nacional del mismo mes** para el pronóstico principal. Son 25 celdas área-mes, o 100 valores al considerar cuatro indicadores; no se convierten en observaciones reales. No se rellenan los meses anteriores al primer ingreso usable. La reducida muestra nacional que pasa los filtros durante la pandemia puede contaminar este sustituto.\n\nSe modelan logaritmos de indicadores positivos; algunas familias usan diferencias y diferencias estacionales. No se aplica Box–Cox ni ajuste por IPC. La exponenciación de un pronóstico en log produce una estimación en nivel que no equivale automáticamente a la esperanza aritmética; el sesgo de retransfomación puede ser relevante. No se eliminan picos por ser incómodos para el ajuste: se señalan pandemia y cambio de marco en 2022 y se compara sensibilidad. El rediseño iniciado en 2021 no queda resuelto por añadir una variable indicadora [@daneempalme; @danerediseno].')
    fig,axes=plt.subplots(3,2,figsize=(9,8),sharex=True)
    for ax,area in zip(axes.flat,AREAS):
        g=monthly[monthly.ciudad==area]
        for ind in IND:ax.plot(g.fecha,g[ind]/1e6,label=LABEL[ind])
        ax.axvspan(pd.Timestamp('2020-03'),pd.Timestamp('2020-07'),color='red',alpha=.12);ax.axvline(pd.Timestamp('2022-01'),ls=':',color='gray');ax.set_title(area);ax.set_ylabel('Millones COP');ax.set_xlabel('Mes')
    axes[0,0].legend(fontsize=7,ncol=2);R.picture(fig,'Series mensuales observadas. Los huecos no son ingresos cero; las líneas no imputan datos.')
    allcv,allmetrics=clasificacion(R,stats)
    R.add('# Modelos de regresión y series de tiempo\n\n## Regresión múltiple del ingreso individual: búsqueda exhaustiva\n\nRespuesta: ingreso mensual en COP, sin transformación logarítmica en este bloque. Se comparan **todas las 511 combinaciones no vacías de nueve variables** en cada dominio: edad, antigüedad, horas, año, sexo, jefatura, estudia, estado civil y educación. Incluye 9 especificaciones de una variable y 502 de dos o más. Una variable categórica entra como bloque completo de dummies, con categoría de referencia; no se cuentan como subconjuntos separados sus niveles. Se mantiene 60/20/20 y se estima imputación y estandarización únicamente en entrenamiento.\n\nSe selecciona cada subconjunto por RMSE en validación, con R² ajustado, SCT, SCR, SCE, AIC y BIC disponibles para todos. Se calcula además MAE para el mejor de cada tamaño. **AIC/BIC no sustituyen el error fuera de muestra.** Las hipótesis OLS incluyen especificación adecuada de la media, ausencia de colinealidad exacta y errores de media condicional cero. Los errores HC1 permiten heterocedasticidad, pero no corrigen por sí solos dependencia de hogar-mes ni incertidumbre de selección de 511 alternativas. Los p-valores son exploratorios, posteriores a selección.')
    for i,a in enumerate(AREAS):
        p=root/'resultados_regresion_final'/str(i);d=loadj(p/'decision_pre_prueba.json');R.add('### '+a+' — combinaciones de variables')
        R.tab(pd.read_csv(p/'mejores_por_tamano.csv')[['k','variables','RMSE','MAE','R2_validacion','R2_ajustado']])
        R.add('El subconjunto OLS elegido es **'+d['variables'].replace('+',', ')+'**. Las 511 filas de búsqueda y los coeficientes HC1 se entregan en resultados_regresion_final/'+str(i)+'/; no se selecciona el subconjunto usando prueba.')
    R.add('## Regresión Ridge\n\nRegularización L2 con alpha=0,1; 1; 10; 100; intercepto sin penalizar. Utiliza todos los bloques predeclarados con numéricos estandarizados. Reduce la inestabilidad de coeficientes correlacionados a cambio de sesgo. Alpha se elige en validación; no se interpreta un coeficiente regularizado como un efecto causal ni se le asigna significancia OLS.')
    R.add('## Regresión Lasso\n\nRegularización L1 con alpha=10,100,1000,10000, máximo 5000 iteraciones e intercepto sin penalizar. Puede anular coeficientes; estos alpha corresponden a la escala de la respuesta en COP, no a una respuesta normalizada. La elección de alpha usa validación. Con predictores correlacionados, el conjunto de coeficientes no nulos puede ser inestable.')
    R.add('## Comparación de regresiones individuales\n\nLa selección final compara el mejor subconjunto OLS y los alpha elegidos de Ridge y Lasso por RMSE de validación. La tabla siguiente muestra prueba independiente. Estos modelos requieren características futuras de cada persona; no se extrapolan automáticamente como una serie mensual de 2026.')
    regtests=[];fig,axes=plt.subplots(3,2,figsize=(9,8))
    for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
        p=root/'resultados_regresion_final'/str(i);q=pd.read_csv(p/'prueba.csv');q.insert(0,'Dominio',a);regtests.append(q);g=pd.read_csv(p/'grafico.csv');ax.scatter(g.observado/1e6,g.pred/1e6,s=3,alpha=.2);ax.set_title(a);ax.set_xlabel('Observado: millones COP');ax.set_ylabel('OLS: millones COP');lim=max(g.observado.quantile(.99),g.pred.quantile(.99))/1e6;ax.set_xlim(0,lim);ax.set_ylim(min(0,g.pred.min()/1e6),lim);ax.plot([0,lim],[0,lim],'r:',lw=.8)
    R.tab(pd.concat(regtests));R.picture(fig,'Regresión individual OLS: observado frente a estimado en una muestra fija de prueba. Ejes recortados al percentil 99 para legibilidad; métricas usan toda la prueba.')
    R.add('Las predicciones negativas, cuando aparecen, evidencian que una regresión lineal en niveles no respeta la positividad del ingreso. No se recortan a cero para mejorar artificialmente el error. R² limitado significa que las características disponibles no explican gran parte de la heterogeneidad; no permite concluir que educación o trayectoria carezcan de relación con el ingreso. Una mejora futura es comparar especificaciones logarítmicas con retransfomación validada.')
    temporal(R)
    annex(R,allcv,allmetrics)
    return R.text()

def temporal(R):
    root=R.root;data=R.csv('resultados_series/series_con_origen.csv',parse_dates=['fecha']);comp=R.csv('resultados_series_final/comparacion.csv');fc=R.csv('resultados_series_final/intervalos.csv',parse_dates=['fecha']);chosen=comp[comp.ganador_final]
    def yy(a,i):return data[(data.ciudad==a)&(data.indicador==i)].set_index('fecha').usado.dropna().asfreq('MS')
    R.add('## Series de tiempo: diseño de evaluación y análisis exploratorio\n\nCada una de las 24 series usa una ventana expansiva: entrenamiento inicial julio de 2008–diciembre de 2021; validación de 12 pasos en 2022, 2023 y 2024, reajustando con información disponible al inicio de cada año. La prueba final son los **12 meses de 2025**, sin incorporar sus valores como rezagos durante el pronóstico recursivo. Se selecciona por RMSE conjunto de los 36 errores de validación. Tras evaluar prueba se reajusta el mismo procedimiento hasta diciembre de 2025 para pronosticar doce meses de 2026.\n\nHay **3 pliegues temporales** de validación, distintos de los 3×2 de clasificación; no se mezclan aleatoriamente meses. La elección de subconjuntos de regresión temporal usa esos tres periodos, por lo que sus errores de validación tienen optimismo de selección. La prueba 2025 se conserva para comparar esa elección fuera de muestra. Los modelos que fallan o no convergen en un pliegue se excluyen de selección; no se sustituyen silenciosamente por otros.')
    desc=R.csv('resultados_series/descriptivas.csv');R.tab(desc[['ciudad','indicador','count','mean','std','min','50%','max']],'Estadísticas de niveles mensuales usados: incluyen los sustitutos nacionales; count no equivale a meses genuinos.')
    ann=R.csv('datos_historicos/anual.csv');fig,axes=plt.subplots(3,2,figsize=(9,7),sharex=True)
    for ax,a in zip(axes.flat,AREAS):
        g=ann[ann.ciudad==a]
        for ind in IND:ax.plot(g.anio,g[ind]/1e6,label=LABEL[ind])
        ax.set_title(a);ax.set_xlabel('Año');ax.set_ylabel('Millones COP')
    axes[0,0].legend(fontsize=7,ncol=2);R.picture(fig,'Resumen anual sobre personas-mes, únicamente en años con doce meses válidos: percentiles de ingresos mensuales agrupados por año, no ingresos anuales ni promedio de percentiles mensuales.')
    eda=R.csv('resultados_series/estacionariedad.csv');R.tab(eda[(eda.indicador=='mediana')][['ciudad','transformacion','ADF_p','KPSS_p','ZA_p','quiebre']],'Pruebas realizadas antes de 2022 sobre log, primera diferencia y diferencia ordinaria más estacional.')
    R.add('ADF contrasta raíz unitaria como hipótesis nula; KPSS contrasta estacionariedad. ADF p<0,05 y KPSS p≥0,05 apoyan estacionariedad; el patrón opuesto sugiere diferenciar. Si ambas rechazan o ninguna rechaza, el resultado es ambiguo y exige considerar potencia, quiebres y especificación. Zivot–Andrews permite un quiebre endógeno, pero su fecha estimada no prueba una causa económica. Las tablas completas incluyen los cuatro indicadores. En KPSS, los valores 0,01 y 0,10 pueden representar límites de la tabla de referencia, no probabilidades exactas.')
    for a in AREAS:
        z=np.log(yy(a,'mediana').loc[:'2021-12']);s=STL(z,period=12,robust=True).fit();fig,axes=plt.subplots(3,2,figsize=(9,7))
        for ax,v,title in zip(axes.flat[:4],[z,s.trend,s.seasonal,s.resid],['Log(COP)','Tendencia STL','Estacionalidad STL','Residuo STL']):ax.plot(v);ax.set_title(title);ax.set_xlabel('Mes');ax.set_ylabel('Unidades log')
        dz=z.diff().dropna();plot_acf(dz,lags=30,ax=axes[2,0],title='ACF de Δlog');plot_pacf(dz,lags=30,method='ywm',ax=axes[2,1],title='PACF de Δlog');R.picture(fig,a+': descomposición robusta de la mediana y correlaciones previas a validación. Periodo estacional 12.')
    R.add('## Especificaciones, supuestos y resultados por modelo temporal\n\nSe estudian 25 procedimientos con la misma ventana. Las figuras por procedimiento muestran la mediana en los seis dominios durante prueba 2025. El anexo por dominio contiene errores de los cuatro indicadores. No se comparan AIC de familias o respuestas transformadas distintas como si fueran intercambiables; el criterio común de selección es error en COP fuera de muestra.')
    descriptions={
      'Ingenuo':'Referencia de persistencia: repite el último nivel observado; no tiene hiperparámetros. Supone que el último mes resume la información relevante.',
      'Estacional':'Referencia estacional: repite los doce meses anteriores. Supone repetición del patrón anual, sin crecimiento adicional.',
      'Deriva':'Referencia con deriva en log: extrapola el cambio promedio entre primer y último mes del entrenamiento. Supone estabilidad de ese crecimiento medio.',
      'SES':'Suavizamiento exponencial simple en log, nivel sin tendencia ni estacionalidad, inicialización estimada y parámetros optimizados. Adecuado si domina un nivel local.',
      'Holt':'Suavizamiento con nivel y tendencia aditiva en log, sin estacionalidad. Extrapolar indefinidamente la pendiente puede producir trayectorias excesivas.',
      'Holt amortiguado':'Holt con tendencia amortiguada en log. El amortiguamiento estimado reduce la contribución futura de la pendiente; no anticipa un choque nuevo.',
      'ETS AAA':'Suavizamiento exponencial con tendencia y estacionalidad aditivas en log, periodo 12 e inicialización estimada. Supone un patrón estacional que evoluciona suavemente.',
      'ETS AAdA':'ETS con tendencia aditiva amortiguada y estacionalidad aditiva en log, periodo 12. La extrapolación desacelera el crecimiento de tendencia.',
      'Regresion exhaustiva':'Regresión dinámica en log: 255 subconjuntos de ocho variables (lag1, lag2, lag12, tendencia, seno, coseno, escalón2022 y pulso2020). Los rezagos futuros se generan recursivamente. Solo 127 subconjuntos resultan identificables en todos los orígenes porque escalón2022 es constante en el primero. Se elige por RMSE de validación; no se introducen todas las variables por defecto.',
      'AR1-ARCH1':'ARCH(1) exploratorio sobre 100×Δlog, media AR(1), errores normales. Modela varianza condicional con un rezago de innovaciones al cuadrado. No participa en la elección del pronóstico final.',
      'AR1-GARCH11':'GARCH(1,1) exploratorio sobre 100×Δlog, media AR(1), errores normales. Modela persistencia de volatilidad mediante innovación y varianza rezagadas. Es habitual en series financieras de alta volatilidad; no es exclusivo de finanzas, pero se excluye del ganador de salarios por decisión explícita del estudio.'}
    for model in MODELS:
        if model in SAR:
            o,s=SAR[model];desc=f'Modelo en log con orden {o} y orden estacional {s}. Se exige estacionariedad/invertibilidad y máximo 150 iteraciones. ACF/PACF y pruebas motivan diferencias ordinarias y estacionales; la malla compara órdenes pequeños para limitar sobreajuste.'
            if model.startswith('SARIMAX'):desc+=' Exógenas: escalón desde enero de 2022 y pulso marzo–julio de 2020. En 2026 se fijan respectivamente en 1 y 0 como escenario conocido; no se estiman futuras políticas o crisis. El escalón no tiene variación antes de 2022, limitando identificación inicial y la lectura causal.'
        elif model.startswith('Auto-malla'):desc='Selección automática auxiliar dentro de tres SARIMA predeclarados con D='+model[-1]+', mediante menor AICc del entrenamiento; el procedimiento completo se evalúa fuera de muestra. No se usa auto_arima como justificación de los órdenes. Se conservan los fallos de convergencia y se excluyen procedimientos incompletos.'
        elif model.startswith('Prophet'):desc='Prophet sobre log(COP): tendencia lineal por tramos, changepoint_prior_scale='+model.split()[1]+', estacionalidad anual Fourier de orden 4; estacionalidad semanal y diaria desactivadas. Ajuste MAP con semilla 78018; los demás valores del constructor son los predeterminados de la versión registrada. Las dos intensidades de regularización se comparan con los mismos tres pliegues anuales. No hay regresores externos ni festivos. [@prophetdiagnostico]'
        elif model.startswith('LSTM'):desc='Red LSTM sobre primeras diferencias de log, normalizadas con media y desviación del entrenamiento: ventana de 12 meses, una capa de 16 unidades, salida lineal de una unidad, Adam con tasa 0,01, pérdida MSE, lote completo, recorte de gradiente a norma 1 y '+model.split()[1]+' épocas. Sin dropout ni parada temprana; las opciones 80/160 épocas se comparan con tres orígenes de validación. Semilla 78018, ejecución determinista CPU y pronóstico recursivo. No se trasladan las 100 neuronas/200 épocas de los ejemplos docentes a una serie corta sin evaluación. [@pytorchlstm]'
        else:desc=descriptions[model]
        R.add('### '+model+'\n\n'+desc)
        paramfile=root/'resultados_series_final/parametros_mediana_todos.csv'
        if paramfile.exists() and not model.startswith('LSTM'):
            allpars=pd.read_csv(paramfile);pars=allpars[allpars.modelo==model].copy()
            if model.startswith('Prophet'):pars=pars[pars.parametro.isin(['k','m','sigma_obs'])]
            elif model in ['SES','Holt','Holt amortiguado','ETS AAA','ETS AAdA']:pars=pars[pars.parametro.str.startswith(('smoothing_','damping_'))]
            elif model=='Estacional':pars=pars[pars.parametro.isin(['ultimos12_log[0]','ultimos12_log[11]'])]
            rows=[]
            for a,g in pars.groupby('ciudad',sort=False):rows.append({'Dominio':a,'Estimaciones':'; '.join(f'{r.parametro}={r.estimacion:.4g}' for _,r in g.iterrows())})
            R.tab(pd.DataFrame(rows),'Estimaciones finales de mediana por dominio. Los coeficientes completos, errores estándar y p-valores disponibles están en parametros_mediana_todos.csv; no son p-valores de redes.')
        q=comp[(comp.modelo==model)&(comp.indicador=='mediana')];R.tab(q[['ciudad','RMSE_validacion','RMSE','MAE','MAPE','MASE','seleccionable']],'Mediana: errores de prueba en COP, MAPE en porcentaje y MASE sin unidades; seleccionable exige convergencia y exclusión de ARCH/GARCH.')
        fig,axes=plt.subplots(3,2,figsize=(9,7),sharex=True)
        for ax,a in zip(axes.flat,AREAS):
            p=pd.read_csv(root/'resultados_series'/f'{TAREAS.index(a)}_mediana/predicciones.csv',parse_dates=['fecha']);p=p[(p.modelo==model)&(p.etapa=='prueba')];ax.plot(p.fecha,p.observado/1e6,'o-',ms=2,label='Observado');ax.plot(p.fecha,p.pred/1e6,'--',label='Pronóstico');ax.set_title(a);ax.set_ylabel('Millones COP');ax.tick_params(axis='x',rotation=30)
        axes[0,0].legend(fontsize=7);R.picture(fig,model+': predicción de la mediana durante 2025, sin actualización con los meses de prueba.')
        rows=[]
        for a in AREAS:
            p=root/'resultados_series'/f'{TAREAS.index(a)}_mediana';di=pd.read_csv(p/'diagnosticos.csv');di=di[di.modelo==model].iloc[0];inf=pd.read_csv(p/'criterios_informacion.csv');inf=inf[inf.modelo==model].iloc[0]
            rows.append(dict(ciudad=a,AIC=inf.get('aic',np.nan),BIC=inf.get('bic',np.nan),LB12=di.LB12,JB=di.JB,ARCH12=di.ARCH12))
        R.tab(pd.DataFrame(rows),'Criterios de información cuando corresponden y p-valores de residuos del ajuste previo a prueba; una celda vacía significa no disponible para ese procedimiento.')
        if model.startswith('LSTM'):
            fig,axes=plt.subplots(3,2,figsize=(9,6))
            for ax,a in zip(axes.flat,AREAS):
                e=loadj(root/'resultados_series'/f'{TAREAS.index(a)}_mediana/extras.json')[model];ax.plot(np.arange(1,len(e['perdidas'])+1),e['perdidas']);ax.set_title(a);ax.set_xlabel('Época');ax.set_ylabel('MSE estandarizado')
            R.picture(fig,model+': pérdida de ajuste final de la mediana. Su descenso no prueba capacidad predictiva; esta se evalúa en los pliegues temporales.')
    R.add('## Aclaración sobre ARCH/GARCH y volatilidad\n\n**GARCH se presenta como ejercicio exploratorio, pero no se tendrá en cuenta para seleccionar el mejor pronóstico**, de acuerdo con el alcance solicitado. Su aplicación habitual es a rendimientos financieros con heterocedasticidad condicional; que una serie salarial tenga picos no basta para justificarlo. Se requieren diagnósticos y una interpretación de la variabilidad de la encuesta. La tabla contrasta ARCH-LM y Ljung–Box de cambios logarítmicos centrados al cuadrado antes de 2022. Cuando no se rechaza ARCH, no se interpreta el ajuste GARCH como evidencia de volatilidad condicional. Los ajustes exploratorios no sustituyen la prueba ni cumplen automáticamente el requisito docente de justificación de GARCH [@archforecast].')
    R.tab(R.csv('resultados_series_final/diagnostico_volatilidad.csv'))
    R.add('### Subconjuntos de la regresión temporal elegidos por serie')
    regs=[]
    for a in AREAS:
        for ind in IND:
            v=loadj(root/'resultados_series'/f'{TAREAS.index(a)}_{ind}/regresion_elegida.json');regs.append(dict(ciudad=a,indicador=ind,variables=', '.join(v['variables']),RMSE_validacion=v['RMSE_validacion'],elegibles=v['elegibles']))
    R.tab(pd.DataFrame(regs),'Las 255 combinaciones y sus tres evaluaciones por serie están en regresion_combinaciones.csv. El tamaño del subconjunto no se fija de antemano.')
    R.add('## Comparación final, diagnóstico y pronósticos de 2026\n\nRMSE penaliza más los errores grandes; MAE mide el desvío absoluto típico; MAPE expresa error relativo y MASE escala MAE con el error estacional ingenuo del entrenamiento. El ganador es el mínimo RMSE de validación entre candidatos elegibles sin ARCH/GARCH, incluso si otro obtiene menor error retrospectivo en 2025. MASE menor que 1 indica superar la escala de referencia dentro de entrenamiento, no necesariamente el menor RMSE de prueba.\n\nLos intervalos **nominales de predicción al 80 % y 95 %** son condicionales al modelo: SARIMA/SARIMAX emplea distribución predictiva gaussiana en log; regresión dinámica y LSTM usan 2000 trayectorias recursivas con bootstrap circular de innovaciones en bloques de 12, manteniendo coeficientes/pesos fijos; Prophet MAP simula tendencia futura y ruido, sin integrar toda la incertidumbre de parámetros estacionales. No incluyen incertidumbre del diseño muestral, del empalme, de la sustitución nacional o de la selección de modelo. Se verifica cobertura en los doce meses de prueba, que es una evaluación pequeña y dependiente temporalmente. Un intervalo estrecho con baja cobertura no debe presentarse como certeza.')
    R.tab(chosen[['ciudad','indicador','modelo','RMSE_validacion','RMSE','MAE','MAPE','MASE']],'24 decisiones finales congeladas antes de prueba, excluyendo ARCH/GARCH.')
    coverage=R.csv('resultados_series_final/cobertura_intervalos.csv');R.tab(coverage,'Cobertura empírica 2025 y ancho medio en COP de los intervalos del modelo elegido.')
    diags=[]
    for _,v in chosen.iterrows():
        q=pd.read_csv(root/'resultados_series'/f'{TAREAS.index(v.ciudad)}_{v.indicador}/diagnosticos.csv');q=q[q.modelo==v.modelo].copy();q.insert(0,'indicador',v.indicador);q.insert(0,'ciudad',v.ciudad);diags.append(q)
    dd=pd.concat(diags);R.tab(dd[['ciudad','indicador','LB12','LB24','JB','ARCH12']],'p-valores de residuos de ajuste hasta 2024: Ljung–Box 12/24, Jarque–Bera y ARCH-LM12.')
    R.add(f'Hay {(dd.LB12<.05).sum()} de 24 ganadores con rechazo de no autocorrelación a 12 rezagos, {(dd.JB<.05).sum()} con rechazo de normalidad y {(dd.ARCH12<.05).sum()} con rechazo de homocedasticidad ARCH-LM al 5 %. No se afirma que todos los residuos sean ruido blanco. Estos rechazos limitan la confianza en los intervalos y motivan mejoras, aunque el modelo tenga menor RMSE de validación. Las pruebas múltiples son exploratorias, sin corrección de multiplicidad.')
    from scipy.stats import probplot
    for a in AREAS:
        fig,axes=plt.subplots(4,3,figsize=(10,10))
        for row,ind in enumerate(IND):
            name=chosen[(chosen.ciudad==a)&(chosen.indicador==ind)].modelo.item();r=pd.read_csv(root/'resultados_series'/f'{TAREAS.index(a)}_{ind}/residuos.csv');r=r[r.modelo==name].residuo.to_numpy();axes[row,0].plot(r,lw=.7);axes[row,0].axhline(0,color='gray',ls=':');axes[row,0].set_title(LABEL[ind]+': residuos',fontsize=8);axes[row,0].set_xlabel('Orden');axes[row,0].set_ylabel('Unidades log');plot_acf(r,lags=24,ax=axes[row,1],title=LABEL[ind]+': ACF');probplot(r,plot=axes[row,2]);axes[row,2].set_title(LABEL[ind]+': Q–Q normal',fontsize=8);axes[row,2].set_xlabel('Cuantil teórico');axes[row,2].set_ylabel('Residuo ordenado')
        R.picture(fig,a+': diagnóstico gráfico de residuos de los cuatro modelos elegidos, ajustados antes de prueba. ACF persistente o colas alejadas de la recta limitan los supuestos predictivos.')
    future=fc[fc.etapa=='pronostico'];cross=future.pivot(index=['ciudad','fecha'],columns='indicador',values='pred');crosses=((cross.p10>cross.mediana)|(cross.mediana>cross.p90)).sum()
    R.add(f'Los pronósticos marginales de cuantiles se verificaron: **{crosses} cruces** P10>P50 o P50>P90 entre los 72 meses-dominio. Se conservan los resultados originales; no se ordenan a posteriori para ocultar un problema de coherencia. Un eventual cruce impediría interpretar esa tripleta como una distribución válida sin reconciliación.')
    for a in AREAS:
        R.add('### '+a+' — pronóstico de los cuatro indicadores')
        fig,axes=plt.subplots(2,2,figsize=(9,6))
        for ax,ind in zip(axes.flat,IND):
            h=yy(a,ind).loc['2024':];g=future[(future.ciudad==a)&(future.indicador==ind)];ax.plot(h.index,h.values/1e6,label='Historia usada');ax.plot(g.fecha,g.pred/1e6,label='2026');ax.fill_between(g.fecha,g.lo95/1e6,g.hi95/1e6,alpha=.15,color='orange',label='95 %');ax.fill_between(g.fecha,g.lo80/1e6,g.hi80/1e6,alpha=.3,color='orange',label='80 %');ax.set_title(LABEL[ind]+' | '+g.modelo.iloc[0],fontsize=8);ax.set_ylabel('Millones COP');ax.tick_params(axis='x',rotation=30)
        axes[0,0].legend(fontsize=7);R.picture(fig,a+': pronóstico enero–diciembre de 2026 y bandas nominales condicionales.')
        for ind in IND:
            g=future[(future.ciudad==a)&(future.indicador==ind)].copy();g['Mes']=g.fecha.dt.strftime('%Y-%m');R.add('#### '+LABEL[ind]);R.tab(g[['Mes','pred','lo80','hi80','lo95','hi95']])
        g=future[(future.ciudad==a)&(future.indicador=='mediana')];last=yy(a,'mediana').iloc[-1];delta=100*(g.pred.iloc[-1]/last-1)
        R.add(f'**Aplicación en {a}:** el modelo de mediana elegido proyecta diciembre de 2026 en {g.pred.iloc[-1]:,.0f} COP, un cambio nominal de {delta:.1f} % frente a diciembre de 2025. El intervalo 95 % para ese mes es [{g.lo95.iloc[-1]:,.0f}; {g.hi95.iloc[-1]:,.0f}] COP. Puede servir como escenario para revisar bandas de remuneración y presupuestos; no es una recomendación de incremento salarial ni poder adquisitivo real. P90 informa sobre la cola superior y suele requerir más cautela por su variabilidad; P10 no equivale al salario mínimo legal.')
    R.add('## Series interrumpidas y escenario sin pandemia\n\nPara cada área y cada indicador se ajusta exclusivamente con enero de 2015–febrero de 2020 un SARIMA (0,1,1)(1,0,0)[12] en log y se proyectan marzo de 2020–diciembre de 2021, con intervalos del 80/95 %. Este contrafactual presupone continuidad del patrón previo. La ventana evita atribuir al COVID el cambio de marco de 2022. No usa la sustitución nacional para aprender el contrafactual urbano.\n\nEn los cinco meses sin dato urbano real, se muestra el sustituto nacional en otro color, **sin calcular una desviación urbana observada**. La diferencia observado–contrafactual después del hueco es descriptiva y mezcla pandemia, composición, otros choques y error del modelo. No puede interpretarse como efecto causal puro ni como pérdida de masa salarial. La suma de brechas mensuales solo se refiere a meses observados y tiene unidades COP-mes del indicador.')
    R.add('### Reconstrucción del hueco urbano: análisis complementario\n\nComo contraste al reemplazo directo solicitado, se conserva un puente log-lineal entre ciudad y nacional con seno/coseno anual, ajustado desde 2012 hasta febrero de 2020 y errores HAC12. Se valida ocultando marzo–julio de 2017, 2018 y 2019. Esta reconstrucción condiciona al ingreso nacional efectivamente observado en pandemia: **no es el contrafactual sin pandemia**. Tampoco se adopta como sustitución principal del pronóstico de 2026.')
    rv=R.csv('resultados_series/validacion_reconstruccion.csv');R.tab(rv.groupby(['ciudad','indicador','metodo'],as_index=False)[['RMSE','MAE']].mean(),'Promedio de errores de los tres ejercicios de ocultamiento de cinco meses; no es RMSE agrupado de todos los errores.')
    rec=R.csv('resultados_series/reconstruccion_2020.csv',parse_dates=['fecha']);fig,axes=plt.subplots(3,2,figsize=(9,7))
    for ax,a in zip(axes.flat,AREAS[:5]):
        g=rec[(rec.ciudad==a)&(rec.indicador=='mediana')];ax.plot(g.fecha,g.reconstruido/1e6,'o-',label='Puente');ax.plot(g.fecha,g.nacional/1e6,'x--',label='Sustituto nacional');ax.fill_between(g.fecha,g.inferior/1e6,g.superior/1e6,alpha=.15);ax.set_title(a);ax.set_ylabel('Millones COP');ax.tick_params(axis='x',rotation=30)
    axes[2,1].axis('off');axes[0,0].legend(fontsize=7);R.picture(fig,'Reconstrucción de la mediana urbana en marzo–julio de 2020, con banda predictiva del puente. Son estimaciones, no observaciones recuperadas.')
    cf=R.csv('resultados_series_final/contrafactuales.csv',parse_dates=['fecha'])
    for a in AREAS:
        fig,axes=plt.subplots(2,2,figsize=(9,7))
        for ax,ind in zip(axes.flat,IND):
            g=cf[(cf.ciudad==a)&(cf.indicador==ind)];ax.plot(g.fecha,g.observado/1e6,'o-',ms=2,label='Observado');ax.plot(g.fecha,g.contrafactual/1e6,'--',label='Sin choque (modelo)');ax.fill_between(g.fecha,g.lo95/1e6,g.hi95/1e6,color='gray',alpha=.18,label='95 %');v=g[g.sustituido];ax.scatter(v.fecha,v.usado/1e6,color='orange',marker='x',label='Nacional sustituto');ax.set_title(LABEL[ind]);ax.set_ylabel('Millones COP');ax.tick_params(axis='x',rotation=30)
        axes[0,0].legend(fontsize=6);R.picture(fig,a+': observado, trayectoria contrafactual y sustitución explícita durante la interrupción.')
        g=cf[cf.ciudad==a];summary=g.groupby('indicador').agg(meses_observados=('observado','count'),brecha_media=('desviacion_observada','mean'),brecha_acumulada=('desviacion_observada','sum')).reset_index();R.tab(summary,'Brecha descriptiva respecto al escenario, calculada solo donde existe dato observado.')
    R.add('### Intervención segmentada nacional y anomalías\n\nComo contraste se conserva la regresión de serie interrumpida nacional: tendencia, pulso y cambio de pendiente, con errores HAC12, junto con el pronóstico SARIMA previo al choque. Se distinguen bandas de incertidumbre del efecto ajustado de intervalos predictivos. Una observación fuera de la banda previa señala incompatibilidad con aquel patrón; no demuestra que la causa sea exclusivamente la pandemia. Los coeficientes nacionales y sus diagnósticos se conservan para auditoría.')
    R.tab(R.csv('resultados_series/intervencion_coeficientes.csv'))
    it=R.csv('resultados_series/intervencion.csv',parse_dates=['fecha']);fig,axes=plt.subplots(2,2,figsize=(9,7))
    for ax,ind in zip(axes.flat,IND):
        g=it[it.indicador==ind];ax.plot(g.fecha,g.observado/1e6,label='Observado');ax.plot(g.fecha,g.ajuste/1e6,label='Intervención ajustada');ax.plot(g.fecha,g.contrafactual/1e6,'--',label='Sin intervención');ax.fill_between(g.fecha,g.pre_inferior/1e6,g.pre_superior/1e6,color='gray',alpha=.15,label='Banda 95 % previa');bad=g[g.anomalia];ax.scatter(bad.fecha,bad.observado/1e6,color='red',s=12,zorder=5,label='Anomalía');ax.axvline(pd.Timestamp('2020-03'),color='red',ls=':');ax.set_title(LABEL[ind]);ax.set_ylabel('Millones COP');ax.tick_params(axis='x',rotation=30)
    axes[0,0].legend(fontsize=5.5);R.picture(fig,'Nacional: intervención segmentada y anomalías respecto a la banda predictiva 95 % del SARIMA ajustado antes del choque. La banda gris pertenece al escenario previo, no al ajuste ITS.')
    R.tab(it[it.fecha>='2020-03'].groupby('indicador',as_index=False).agg(meses=('fecha','size'),meses_atipicos=('anomalia','sum')),'Conteo de meses fuera de la banda previa. Ser atípico no identifica por sí solo la causa ni implica un dato incorrecto.')
    pf=root/'resultados_ml_final'/str(AREAS.index(FINAL));wf=loadj(pf/'decision_pre_prueba.json')['ganador'];qf=pd.read_csv(pf/'prueba.csv').set_index('modelo');tf=pd.read_csv(pf/'tasas_clase.csv');tf=tf[tf.regla=='costos'].set_index('clase')
    R.add('## Conclusiones aplicativas y limitaciones integradas\n\n'+f'**Clasificación.** El modelo final es un único {wf} nacional con la ciudad como variable, cortes P10/P50/P90 por ciudad y año calculados en entrenamiento y una regla de decisión por matriz de costos que declara más caro dejar por fuera a una persona de C1. En prueba alcanza {1000*qf.loc[wf,"costo_regla"]:.0f} unidades de costo por mil personas, frente a {1000*qf.loc["Constante de costo","costo_regla"]:.0f} de la mejor clase constante, con TPR de C1 de {tf.loc["C1","TPR"]:.2f} y VPP de {tf.loc["C1","VPP"]:.2f}. Es el modelo que describe la ficha y el único que se despliega; los modelos por área quedan como comparación pareada. Las tasas por grupo muestran brechas que deben vigilarse antes de cualquier uso.\n\n'+'La clasificación responde quién presenta un perfil asociado a cada tramo relativo; la regresión individual aproxima montos; el pronóstico temporal describe cómo podría evolucionar un indicador agregado. Sus errores no son intercambiables. Las matrices muestran errores suficientes para descartar una asignación automática de ingresos individuales. Los perfiles y ejemplos permiten revisar concretamente quién es confundido, sin inventar personas ni atribuirles una causa salarial.\n\nUna organización puede usar el estudio para diseñar hipótesis de segmentación y contenidos informativos, revisar presupuestos agregados y simular escenarios de remuneración por ciudad. Para enviar correos a clientes se necesita una base propia autorizada, validar que la población y variables sean comparables y comprobar el rendimiento de una campaña; GEIH no permite identificar ni contactar encuestados. No se ha medido retorno comercial ni impacto causal del uso del modelo.\n\nEn series, el mejor modelo cambia entre indicadores y áreas; no hay una familia ganadora universal. Los intervalos y sus coberturas son más informativos que una cifra puntual aislada. La sustitución urbana por nacional permite ejecutar el pronóstico, pero impone igualdad artificial durante cinco meses. El escenario sin pandemia es útil para discutir sensibilidad y discontinuidad, no para cuantificar causalmente daños.\n\nPersisten límites: muestra no ponderada y restringida, menos de 240 observaciones, fecha original de descarga incompleta, ingreso no recuperado al comienzo, cambios de medición en 2020/2022, empates en cuantiles, dependencia intrahogar, incertidumbre por búsqueda de modelos y parámetros, predicciones nominales sin IPC y desplazamiento de población futura. Se propone recuperar módulos históricos faltantes, validar empalmes, ponderar como análisis adicional, evaluar cobertura con más orígenes, añadir auditoría de equidad y recalibrar en una muestra reciente antes de uso operativo.')

def annex(R,allcv,allmetrics):
    root=R.root
    R.add('# Anexos de parámetros, comparaciones y reproducción\n\n## Pliegues externos de clasificación\n\nEstos son resultados por pliegue, calculados sobre las mismas particiones dentro de cada dominio. La configuración puede cambiar entre pliegues porque se selecciona únicamente con su búsqueda interna. Los archivos cv_busqueda.csv conservan también los dos pliegues internos y todos los candidatos, y pliegues.parquet permite reconstruir la asignación.')
    for a in AREAS:
        q=allcv[allcv.Dominio==ETIQUETA[a]];R.add('### '+ETIQUETA[a])
        R.tab(pd.DataFrame({'Modelo':q.modelo.values,'Pliegue':q.pliegue.values,'Configuración':q.param.values,'F1 macro':q.f1_macro.values,'Costo regla por mil':mil(q.costo_regla)}))
    R.add('## Curvas ROC y precisión–recobrado por modelo\n\nSe muestran curvas uno contra el resto para cada clase. ROC compara sensibilidad y falsos positivos; PR hace visible la precisión al aumentar cobertura. Las curvas se reducen a hasta 250 puntos solo para dibujar; AUC usa toda la prueba. Los márgenes SVM sirven para ordenar, sin interpretarlos como probabilidades calibradas.')
    for model in GRIDS:
        for kind in ['ROC','PR']:
            fig,axes=plt.subplots(3,2,figsize=(9,7))
            for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
                q=pd.read_csv(root/'resultados_ml_final'/str(i)/'curvas.csv');q=q[(q.modelo==model)&(q.tipo==kind)]
                for c,g in q.groupby('clase'):ax.plot(g.x,g.y,label='C'+str(int(c)+1))
                if kind=='ROC':ax.plot([0,1],[0,1],':',color='gray')
                ax.set_title(ETIQUETA[a]);ax.set_xlabel('Falsos positivos' if kind=='ROC' else 'Recobrado');ax.set_ylabel('Sensibilidad' if kind=='ROC' else 'Precisión');ax.set_xlim(0,1);ax.set_ylim(0,1)
            axes[0,0].legend(fontsize=7);R.picture(fig,model+': '+kind+' por clase y dominio.')
    R.add('## Errores temporales de todos los modelos y todos los indicadores')
    q=R.csv('resultados_series_final/comparacion.csv')
    for a in AREAS:
        R.add('### '+a)
        for ind in IND:
            R.add('#### '+LABEL[ind]);R.tab(q[(q.ciudad==a)&(q.indicador==ind)][['modelo','RMSE_validacion','RMSE','MAE','MAPE','MASE','ganador_final']])
    R.add('## Coeficientes estimados de los pronósticos elegidos\n\nLos errores estándar y p-valores que siguen corresponden a los modelos que los proporcionan; no se inventa significancia para redes neuronales, Prophet MAP o pesos regularizados. Las cifras son posteriores a selección y no se interpretan como evidencia causal. Para parámetros fijos, arquitectura, optimización y semillas se remite a las subsecciones por modelo y a los scripts incluidos.')
    q=R.csv('resultados_series_final/parametros.csv');q=q[q.etapa=='pronostico']
    for a in AREAS:
        g=q[q.ciudad==a];R.add('### '+a);R.tab(g[['indicador','modelo','parametro','estimacion','se','p']])
    R.add('## Estacionariedad de todos los indicadores')
    q=R.csv('resultados_series/estacionariedad.csv')
    for a in AREAS:R.add('### '+a);R.tab(q[q.ciudad==a].drop(columns='ciudad'))
    R.add('## Tasas por grupo con la clase de máxima probabilidad y desempeño por subgrupo\n\nComo referencia de la sección de tasas por grupo, se repiten TPR, FPR y VPP del modelo final con la clase de máxima probabilidad, que no es la regla desplegada. Diferencias de tasas no bastan para diagnosticar equidad: prevalencia, error por clase, incertidumbre y variables proxy necesitan análisis adicional. Sexo se utilizó como predictor; no se acredita neutralidad ni cumplimiento de una política de exclusión de atributos protegidos.')
    pf=root/'resultados_ml_final'/str(AREAS.index(FINAL));tg=pd.read_csv(pf/'tasas_grupo.csv');tg=tg[tg.regla=='argmax']
    for var in ['sexo','grupo_edad','educacion','ciudad']:
        g=tg[tg.variable==var].copy();g['orden']=g.grupo.map({v:i for i,v in enumerate(ORDEN_GRUPOS[var])});g=g.sort_values(['orden','clase'])
        R.add('### '+VARIABLE_TXT[var]+' (máxima probabilidad)');R.tab(pd.DataFrame({'Grupo':g.grupo.values,'Clase':g.clase.values,'n':g.n.values,'Prevalencia':g.prevalencia.values,'Asignados':g.predichos.values,'TPR':g.TPR.values,'FPR':g.FPR.values,'VPP':g.VPP.values}))
    R.add('Los registros sin sexo o educación observados no integran el desglose del atributo respectivo; sí reciben predicción mediante la imputación aprendida en entrenamiento. Las tablas siguientes resumen, para cada dominio, F1 macro, exactitud balanceada, error ordinal y costo por mil del modelo elegido con la regla de costos en cada subgrupo.')
    for i,a in enumerate(AREAS):
        R.add('### '+ETIQUETA[a]+' — subgrupos');q=pd.read_csv(root/'resultados_ml_final'/str(i)/'subgrupos.csv')
        R.tab(pd.DataFrame({'Variable':q.variable.map(VARIABLE_TXT).values,'Grupo':q.grupo.values,'n':q.n.values,'F1 macro':q.f1_macro.values,'Exactitud balanceada':q.balanceada.values,'Error ordinal':q.error_ordinal.values,'Costo por mil':mil(q.costo)}))
    R.add('## Reproducibilidad, entrega y correspondencia con las rúbricas\n\nEl archivo LEEME_PROFESOR.md describe instalación y ejecución desde cualquier carpeta. El QMD genera el índice automáticamente con toc: true y toc-depth: 3; basta añadir encabezados Markdown. El código ejecuta primero cálculos con semilla fija y cache verificada, luego las tablas/figuras y Quarto. Los parámetros completos de los clasificadores están en parametros_0.json a parametros_6.json de cada dominio, además de los modelos serializados y los datos analíticos. La matriz de costos y la regla de decisión están en decision_costos.py; el modelo desplegado, con su preprocesador, cortes, configuración y ficha, está en modelo_final/ y se usa con cargar_y_predecir.py. El archivo de versiones registra el entorno real, no solo dependencias mínimas.\n\nLa entrega incluye datos analíticos, partición, bibliografía, estilo CSL, logo, scripts, QMD, PDF y el modelo final empaquetado. El Parquet conserva tipos y agiliza la reproducción. Los archivos originales de la encuesta no se confunden con estas bases filtradas. El profesor puede revisar el modelo final y el PDF o recalcular; el ajuste nacional y la búsqueda temporal requieren tiempo y memoria (la variable GEIH_NJOBS fija los núcleos sin cambiar los resultados).\n\n**Adaptaciones explícitas:** el estudio es multiclase, con un único modelo final nacional que usa la ciudad como variable y cinco modelos de comparación por área; usa CV 3×2, regla de decisión por matriz de costos, SVM lineal y un informe integrado más extenso que el límite del primer artículo. No acredita la excepción de 240 meses, ni una aplicación comercial desplegada, ni inferencia causal. La comparación exhaustiva es dentro del conjunto de variables declarado, no de todas las variables posibles de GEIH. Las conclusiones y decisiones deben ser revisadas por los autores antes de entregar. Se utilizó asistencia de IA para programación, organización y redacción, y se contrastan los resultados con archivos calculados; no se simulan aprobación del docente ni validaciones externas inexistentes.\n\n# Referencias {.unnumbered}\n\n::: {#refs}\n:::')
    import importlib.metadata as metadata
    # Insertar versiones antes de la bibliografía, conservando esta al final.
    last=R.parts.pop();pre,refs=last.split('# Referencias',1);R.add(pre)
    R.tab(pd.DataFrame([{'Paquete':p,'Versión':metadata.version(p)} for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','arch','prophet','torch','matplotlib','pyarrow','dane-data']]),'Versiones efectivas del entorno que genera este documento; semilla 78018. Véanse entorno_final.json y pip_freeze_final.txt.')
    R.add('# Referencias'+refs)


# ---------------------------------------------------------------------------
# Clasificación: modelo final nacional con ciudad, regla de costos, tasas por
# grupo, ficha y comparación pareada con los modelos de las cinco áreas.
# ---------------------------------------------------------------------------
ORDEN_GRUPOS={'sexo':['Hombre','Mujer'],'grupo_edad':dc.EDADES,'educacion':['Primaria o menos','Secundaria','Media','Superior'],'ciudad':CIUDADES+[RESTO]}
VARIABLE_TXT={'sexo':'Sexo','grupo_edad':'Edad','educacion':'Educación','ciudad':'Ciudad'}

def mil(v):
    """Costo medio por persona expresado por cada mil personas (entero)."""
    return (1000*pd.Series(np.asarray(v,dtype=float))).round().astype('Int64').array

def miles(v):
    return f'{int(v):,}'.replace(',',' ')

def entero(v):
    return pd.array([None if (x is None or (isinstance(x,float) and np.isnan(x))) else int(x) for x in v],dtype='Int64')

def bootstrap_hogares(hogar,real,pred,reps=300,seed=78018):
    """IC 95 % por remuestreo de hogares-mes de prueba (modelo, cortes y regla fijos): costo medio y F1 macro."""
    codes=pd.factorize(np.asarray(hogar))[0];ng=codes.max()+1
    blocks=np.zeros((ng,16),dtype=np.int32);np.add.at(blocks,(codes,np.asarray(real,dtype=int)*4+np.asarray(pred,dtype=int)),1)
    rng=np.random.default_rng(seed);C=dc.COSTOS.reshape(-1);out=[]
    for _ in range(reps):
        m=blocks[rng.integers(0,ng,ng)].sum(axis=0);mat=m.reshape(4,4);tp=np.diag(mat).astype(float)
        rec=np.divide(tp,mat.sum(1),out=np.zeros(4),where=mat.sum(1)>0);prec=np.divide(tp,mat.sum(0),out=np.zeros(4),where=mat.sum(0)>0)
        f=np.divide(2*prec*rec,prec+rec,out=np.zeros(4),where=(prec+rec)>0);out.append(((m*C).sum()/m.sum(),f.mean()))
    return np.quantile(np.array(out),[.025,.975],axis=0)

def bootstrap_diferencia(hogar,real,pa,pb,reps=300,seed=78018):
    """IC 95 % de la diferencia de costo medio (b − a) sobre las mismas personas, remuestreando hogares-mes."""
    codes=pd.factorize(np.asarray(hogar))[0];ng=codes.max()+1;real=np.asarray(real,dtype=int)
    ca=np.bincount(codes,weights=dc.COSTOS[real,np.asarray(pa,dtype=int)],minlength=ng);cb=np.bincount(codes,weights=dc.COSTOS[real,np.asarray(pb,dtype=int)],minlength=ng);nn=np.bincount(codes,minlength=ng)
    rng=np.random.default_rng(seed);d=[]
    for _ in range(reps):
        ix=rng.integers(0,ng,ng);d.append((cb[ix].sum()-ca[ix].sum())/nn[ix].sum())
    return np.quantile(d,[.025,.975])

def clasificacion(R,stats):
    root=R.root;P=lambda i:root/'resultados_ml_final'/str(i);pf=P(AREAS.index(FINAL))
    R.add('# Modelos de machine learning\n\n## Pregunta, modelo final y diseño\n\n'
        '**Pregunta.** ¿Qué tan bien clasifica un único modelo nacional, con la ciudad como variable, el ingreso laboral de un asalariado en cuatro tramos C1–C4, definidos por P10, P50 y P90 calculados en entrenamiento por ciudad y año? '
        '**Modelo final.** Se ajusta en el dominio nacional con nueve predictores: edad, antigüedad, horas, sexo, jefatura del hogar, estudia, estado civil, educación y **ciudad**, con seis categorías (Medellín AM, Barranquilla AM, Bogotá, Bucaramanga AM, Cali AM y Resto del país). '
        'Es el único modelo que se despliega (carpeta modelo_final/ y cargar_y_predecir.py) y el que describe la ficha. **Comparación.** Cada una de las cinco áreas conserva un modelo propio con el mismo protocolo y sin la variable ciudad, constante dentro del área; esos modelos no se despliegan.\n\n'
        'Las clases se definen con cortes P10, P50 y P90 **por ciudad y año**, estimados solo con el conjunto de ajuste de cada etapa (subpliegue interno, pliegue externo o entrenamiento final) y aplicados congelados al conjunto evaluado. Así, una persona de Bogotá se compara con el ingreso de Bogotá en su año y una del Resto del país con el resto del país. '
        'El ingreso solo define la respuesta; nunca entra como predictor, y el año tampoco. Imputación por mediana y estandarización en numéricas; moda y one-hot en categóricas; toda transformación se aprende dentro de cada ajuste [@sklearnfugas].\n\n'
        'Se mantiene 60 % entrenamiento, 20 % validación y 20 % prueba por grupos de hogar-mes, con semilla 78018. **La partición nacional respeta la de las cinco áreas:** los hogares de cada área conservan su conjunto y los del Resto del país se reparten 60/20/20 dentro de cada año. '
        'Por eso la prueba del modelo final restringida a un área contiene exactamente las mismas personas, con las mismas clases, que la prueba del modelo de esa área, y ningún hogar de prueba de un área entra al entrenamiento nacional. Dentro del 60 % se aplican **3 pliegues externos × 2 internos** agrupados por hogar-mes y compartidos por todas las familias.\n\n'
        '**Criterio de selección.** Como el modelo se usa con la regla de costos, las familias con probabilidades (logística, árbol, bosque, KNN y red MLP) eligen hiperparámetros por **menor costo esperado** en la búsqueda interna. Los SVM, cuyos márgenes no son probabilidades, se eligen por F1 macro y quedan solo como comparación. '
        'La configuración final es la moda de las elegidas en los pliegues externos (desempate por el criterio interno medio); se reajusta en el 60 % y la familia final es la de **menor costo esperado en la validación externa del 20 %**, con desempate por F1 macro. La decisión se guarda antes de examinar prueba.\n\n'
        'Los hogares-mes de prueba no aparecen en entrenamiento ni validación; esto no garantiza independencia de una persona que reaparezca en otro mes. La partición aleatoria dentro de años mide generalización a perfiles contemporáneos, no traslado temporal a 2026. La búsqueda cubre una malla finita. Sexo permanece entre los predictores: por eso se informan tasas por grupo y no se propone una decisión individual automática.')
    R.tab(pd.DataFrame([{'Dominio':ETIQUETA[s['area']],'Papel':'Final (se despliega)' if s['area']==FINAL else 'Comparación','Predictores':len(s['features']),'Entrenamiento':s['entrenamiento'],'Validación':s['validacion'],'Prueba':s['prueba'],'Familia elegida':s['ganador']} for s in stats]),'Modelo final y modelos de comparación. En cada área, la prueba del modelo final contiene exactamente las personas de prueba del modelo de esa área.')
    R.tab(pd.DataFrame([{'Modelo':m,'Malla':json.dumps(v,ensure_ascii=False)} for m,v in GRIDS.items()]),'Mallas de hiperparámetros, idénticas en todos los dominios.')
    regla(R,pf)
    metrics=[];cvs=[];case_rows=[]
    cols=['id_persona','anio','edad','educacion','antiguedad','horas','sexo','jefe','estudia','estado_civil']
    bases={'c':pd.read_parquet(root/'resultados_historicos/base_individual.parquet',columns=cols),'n':pd.read_parquet(root/'datos_ml_nacional/base_individual.parquet',columns=cols)}
    for i,a in enumerate(AREAS):
        p=P(i);q=pd.read_csv(p/'prueba.csv');q.insert(0,'Dominio',ETIQUETA[a]);metrics.append(q)
        q=pd.read_csv(p/'cv_externa.csv');q.insert(0,'Dominio',ETIQUETA[a]);cvs.append(q)
        ex=pd.read_parquet(p/'predicciones.parquet').groupby('modelo',sort=False).head(2)
        ex=ex.merge(bases['n' if a==FINAL else 'c'],on='id_persona',validate='many_to_one').merge(pd.read_parquet(p/'particion.parquet',columns=['id_persona','ciudad']),on='id_persona',validate='many_to_one')
        ex['Dominio']=ETIQUETA[a];ex['Caso']=ex.groupby('modelo').cumcount()+1;ex['real']+=1;ex['predicha']+=1;ex['predicha_regla']=(ex.predicha_regla.where(ex.predicha_regla>=0)+1).astype('Int64');case_rows.append(ex)
    del bases
    case_rows=pd.concat(case_rows,ignore_index=True);case_rows.drop(columns='id_persona').to_csv(root/'resultados_ml_final/ejemplos_todos_modelos.csv',index=False)
    allmetrics=pd.concat(metrics,ignore_index=True);allcv=pd.concat(cvs,ignore_index=True)
    familias(R,allmetrics,allcv,case_rows)
    agrupamiento(R)
    modelo_final(R,allmetrics)
    tasas_grupo(R)
    ficha(R)
    comparacion_areas(R,allmetrics)
    return allcv,allmetrics

def regla(R,pf):
    R.add('## Regla de decisión: matriz de costos\n\n'
        'La versión binaria usaba un umbral por costo (falso negativo = 3, falso positivo = 1; umbral p* = 1/(1+3) = 0,25). Con cuatro clases no hay un único umbral, así que la decisión se toma con una **matriz de costos** que dice qué error es más caro. '
        'El uso previsto es priorizar el contenido de una campaña informativa de orientación laboral y financiera cuando el ingreso no se declara; C1 (ingreso ≤ P10 de su ciudad y año) es el grupo prioritario.\n\n'
        '- **El error más caro es dejar por fuera a una persona de C1**, es decir, asignarle un tramo superior: cuesta 3 por cada tramo de distancia (3, 6 o 9), igual que el falso negativo de la versión binaria, porque esa persona pierde la orientación que le corresponde.\n'
        '- **Los demás errores cuestan la distancia entre tramos** (1, 2 o 3) en ambos sentidos: confundir tramos vecinos es menos grave que confundir extremos. Incluir por error en C1 a una persona de C2 cuesta 1; a una de C4, 3.')
    m=pd.DataFrame(dc.COSTOS.astype(int),columns=[f'Asignada {c}' for c in dc.NOMBRES]);m.insert(0,'Clase real',dc.NOMBRES)
    R.tab(m,'Matriz de costos (unidades hipotéticas). Fila = clase real; columna = clase asignada; la diagonal no cuesta. La fila C1 contiene los errores más caros.')
    R.add('Para cada persona el modelo estima p = (p1, p2, p3, p4) y se asigna la clase k que minimiza el costo esperado $\\sum_j p_j\\,C_{jk}$; en un empate se toma el tramo menor. Con esta matriz la regla asigna **C1 si y solo si P(C1 | x) ≥ 0,25**, exactamente el umbral binario, y entre C2, C3 y C4 decide el costo ordinal esperado. '
        'La matriz se fijó por la aplicación antes de mirar validación o prueba; no se ajusta con los datos. Las unidades son hipotéticas: no son pesos ni daño social medido. La misma matriz sirve para elegir hiperparámetros y familia, para decidir en prueba y en el despliegue.\n\n'
        'La sensibilidad siguiente varía solo el costo de excluir a una persona de C1 (1 significa que todos los errores cuestan la distancia), con las probabilidades de validación del modelo final. Al aumentarlo, la regla asigna más personas a C1: sube la TPR de C1 y baja su VPP. No se usa para escoger la matriz.')
    s=pd.read_csv(pf/'sensibilidad_costos.csv')
    R.tab(pd.DataFrame({'Costo de excluir a C1':s.prioridad.astype(int),'Umbral P(C1)':s.tau,'Asignados C1':s.asignados_C1,'Asignados C2':s.asignados_C2,'Asignados C3':s.asignados_C3,'Asignados C4':s.asignados_C4,'TPR C1':s.TPR_C1,'VPP C1':s.VPP_C1,'F1 macro':s.f1_macro,'Costo por mil (matriz principal)':mil(s.costo_principal)}),'Sensibilidad en validación del modelo final: proporción asignada a cada clase, TPR y VPP de C1, F1 macro y costo evaluado siempre con la matriz principal (costo 3).')

def familias(R,allmetrics,allcv,case_rows):
    root=R.root
    for mi,model in enumerate(GRIDS):
        prob=model in dc.PROBABILISTICOS
        R.add('## '+model+'\n\n'+MODEL_TEXT[model]+(' Produce probabilidades, por lo que admite la regla de costos; sus hiperparámetros se eligen por costo esperado.' if prob else ' Sus márgenes no son probabilidades: se elige por F1 macro, se evalúa con la clase de mayor margen y no puede desplegarse con la regla de costos.'))
        pars=[]
        for i,a in enumerate(AREAS):
            q=loadj(root/'resultados_ml_final'/str(i)/f'parametros_{mi}.json');pars.append({'Dominio':ETIQUETA[a],'Configuración':json.dumps(q['elegidos']),'n ajuste':q['n_ajuste'],'Época elegida':q['epoca'],'Componentes':q['componentes']})
        pars=pd.DataFrame(pars);pars['Época elegida']=entero(pars['Época elegida']);pars['Componentes']=entero(pars['Componentes'])
        R.tab(pars,'Configuración elegida en cada dominio; todos los valores del estimador están en parametros_'+str(mi)+'.json.')
        cv=allcv[allcv.modelo==model].groupby('Dominio',sort=False).agg(F1m=('f1_macro','mean'),F1s=('f1_macro','std'),cc=('costo_regla','mean')).reset_index()
        q=allmetrics[allmetrics.modelo==model]
        if prob:
            t=pd.DataFrame({'Dominio':q.Dominio.values,'Costo regla por mil':mil(q.costo_regla),'F1 regla':q.f1_macro_regla.values,'Exactitud regla':q.exactitud_regla.values,'F1 máx. prob.':q.f1_macro.values,'AUC macro':q.auc_macro.values})
            t=t.merge(pd.DataFrame({'Dominio':cv.Dominio,'Costo CV por mil':mil(cv.cc),'F1 CV media':cv.F1m,'F1 CV DE':cv.F1s}),on='Dominio')
        else:
            t=pd.DataFrame({'Dominio':q.Dominio.values,'F1 máx. margen':q.f1_macro.values,'Exactitud':q.exactitud.values,'Costo máx. margen por mil':mil(q.costo_argmax),'AUC macro':q.auc_macro.values}).merge(pd.DataFrame({'Dominio':cv.Dominio,'F1 CV media':cv.F1m,'F1 CV DE':cv.F1s}),on='Dominio')
        R.tab(t,'Resultados en prueba (20 %) y media/DE entre los tres pliegues externos; la DE no es un intervalo de confianza. Costos en unidades hipotéticas por cada mil personas.')
        e=case_rows[case_rows.modelo==model]
        R.tab(pd.DataFrame({'Dominio':e.Dominio,'Caso':e.Caso,'Ciudad':e.ciudad,'Año':e.anio,'Edad':e.edad,'Educación':e.educacion,'Real':e.real,'Máx. prob.':e.predicha,'Regla':e.predicha_regla}),'Predicciones concretas: los mismos dos primeros casos de prueba de cada dominio para todos los modelos, sin escoger solo aciertos. Clases 1 a 4; la columna Regla es la clase asignada con la matriz de costos (— si el modelo no da probabilidades).')
        fig,axes=plt.subplots(3,2,figsize=(8,9));kind='costos' if prob else 'argmax'
        for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
            cm=pd.read_csv(root/'resultados_ml_final'/str(i)/'confusiones.csv');cm=cm[(cm.modelo==model)&(cm.regla==kind)].pivot(index='real',columns='predicha',values='n').reindex(index=range(4),columns=range(4)).to_numpy();rate=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
            ax.imshow(rate,vmin=0,vmax=1,cmap='Blues')
            for r in range(4):
                for c in range(4):ax.text(c,r,f'{cm[r,c]:,}\n{rate[r,c]:.0%}',ha='center',va='center',fontsize=8,color='white' if rate[r,c]>.55 else 'black')
            ax.set_xticks(range(4),dc.NOMBRES);ax.set_yticks(range(4),dc.NOMBRES);ax.set_xlabel('Clase asignada');ax.set_ylabel('Clase real');ax.set_title(ETIQUETA[a])
        R.picture(fig,model+(': matrices de confusión en prueba con la regla de costos.' if prob else ': matrices de confusión en prueba con la clase de mayor margen (sin regla de costos).')+' Conteos y porcentajes normalizados por fila; la diagonal indica aciertos.')
        if model=='Red MLP':
            fig,axes=plt.subplots(3,2,figsize=(9,7))
            for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
                c=pd.read_csv(root/'resultados_ml_final'/str(i)/'red_aprendizaje.csv');ax.plot(c.epoca,c.perdida_entreno,label='Ajuste');ax.plot(c.epoca,c.perdida_parada,label='Parada interna');ax.set_title(ETIQUETA[a]);ax.set_xlabel('Época');ax.set_ylabel('Log-loss')
            axes[0,0].legend();R.picture(fig,'Red MLP: curvas de entrenamiento y parada temprana, sin utilizar prueba.')
        if prob:
            R.add(f'**Interpretación de {model}.** Con la regla de costos, su costo en prueba va de {1000*q.costo_regla.min():.0f} a {1000*q.costo_regla.max():.0f} por mil según el dominio, frente a {1000*q.costo_argmax.min():.0f}–{1000*q.costo_argmax.max():.0f} si se asignara la clase de máxima probabilidad. '
                f'La F1 macro de la regla va de {q.f1_macro_regla.min():.3f} a {q.f1_macro_regla.max():.3f} (máxima probabilidad: {q.f1_macro.min():.3f}–{q.f1_macro.max():.3f}): la regla desplaza casos hacia C1 cuando P(C1) ≥ 0,25 y acepta más inclusiones erróneas para no dejar por fuera al grupo prioritario. Las curvas ROC y precisión–recobrado están en el anexo.')
        else:
            R.add(f'**Interpretación de {model}.** Su F1 macro de prueba varía entre {q.f1_macro.min():.3f} y {q.f1_macro.max():.3f} según el dominio. Como sus márgenes no son probabilidades calibradas, no se le aplica la regla de costos y no es elegible como modelo final; se conserva como referencia de una frontera lineal.')

def agrupamiento(R):
    root=R.root
    R.add('## ACP y agrupamiento K-means\n\nEl ACP descriptivo y K-means se ajustan con los mismos predictores transformados del entrenamiento de cada dominio, sin el ingreso ni la clase; en el modelo final la representación incluye la ciudad. K-means compara k=2,3,4,5,6, n_init=3, máximo 150 iteraciones y semilla 78018; se elige mayor silhouette de validación sobre hasta 2000 casos con muestra fija. El gráfico muestra hasta 5000 casos de prueba proyectados en las dos primeras componentes. Distancias euclídeas y codificación one-hot ponderan categorías según su representación; los conglomerados no son estratos salariales ni tienen una matriz de confusión de clases verdaderas.')
    fig,axes=plt.subplots(3,2,figsize=(9,8));rows=[]
    for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
        p=root/'resultados_ml_final'/str(i);q=pd.read_csv(p/'clusters_grafico.csv');ax.scatter(q.cp1,q.cp2,c=q.cluster,s=3,alpha=.3,cmap='tab10');ax.set_title(ETIQUETA[a]);ax.set_xlabel('Componente 1');ax.set_ylabel('Componente 2');k=pd.read_csv(p/'clusters.csv');k.insert(0,'Dominio',ETIQUETA[a]);rows.append(k)
    R.picture(fig,'Perfiles agrupados por K-means, visualizados mediante ACP. Los colores identifican grupos, sin orden salarial.')
    R.tab(pd.concat(rows),'Todas las alternativas de k; silhouette mayor indica separación geométrica más marcada, no utilidad económica demostrada.')
    fig,axes=plt.subplots(3,2,figsize=(9,7))
    for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
        q=pd.read_csv(root/'resultados_ml_final'/str(i)/'acp.csv');ax.plot(q.componente,q.varianza.cumsum(),marker='.');ax.axhline(.95,color='gray',ls=':');ax.set_title(ETIQUETA[a]);ax.set_xlabel('Número de componentes');ax.set_ylabel('Varianza acumulada')
    R.picture(fig,'ACP: varianza acumulada del entrenamiento por dominio.')
    R.add('### Perfiles originales y estabilidad de los conglomerados\n\nSe repite K-means con el k elegido y semillas 78019 y 78020, manteniendo el entrenamiento completo y el preprocesamiento. Se compara con la solución de semilla 78018 mediante Rand ajustado (ARI) sobre una muestra fija de hasta 5000 casos de validación. ARI=1 indica la misma partición salvo el nombre de los grupos. Esto audita **inicialización**, no estabilidad frente a otra población ni a una muestra bootstrap. Los ingresos y clases de prueba (con los cortes por ciudad y año) solo se emplean después para describir los grupos.')
    stability=[];mix=[]
    for i,a in enumerate(AREAS):
        p=root/'resultados_ml_final'/str(i);s=pd.read_csv(p/'clusters_estabilidad.csv');s.insert(0,'Dominio',ETIQUETA[a]);stability.append(s);g=pd.read_csv(p/'clusters_perfiles.csv');R.add('#### '+ETIQUETA[a]);R.tab(g[['cluster','n','edad_mediana','antiguedad_mediana','horas_medianas','ingreso_mediano','educacion_modal']]);g.insert(0,'Dominio',ETIQUETA[a]);mix.append(g[['Dominio','cluster','prop_C1','prop_C2','prop_C3','prop_C4']])
    R.tab(pd.concat(stability),'Estabilidad de inicialización. Los grupos no se nombran automáticamente como clases económicas.')
    R.tab(pd.concat(mix),'Composición salarial observada de cada conglomerado en prueba; proporciones entre 0 y 1. La presencia de varias clases en un grupo muestra que agrupar perfiles y clasificar ingresos son tareas diferentes.')

def modelo_final(R,allmetrics):
    root=R.root;pf=root/'resultados_ml_final'/str(AREAS.index(FINAL));LF=ETIQUETA[FINAL]
    dec=loadj(pf/'decision_pre_prueba.json');w=dec['ganador'];q=allmetrics[allmetrics.Dominio==LF].set_index('modelo')
    r=q.loc[w];cte=q.loc['Constante de costo']
    tc=pd.read_csv(pf/'tasas_clase.csv');a_=tc[tc.regla=='costos'].set_index('clase');b_=tc[tc.regla=='argmax'].set_index('clase')
    k=int(np.argmin([abs(float(a_.prevalencia.values@dc.COSTOS[:,j])-cte.costo_regla) for j in range(4)]))
    pred=pd.read_parquet(pf/'predicciones.parquet');pw=pred[pred.modelo==w].merge(pd.read_parquet(pf/'particion.parquet',columns=['id_persona','id_hogar']),on='id_persona',validate='one_to_one')
    ci=bootstrap_hogares(pw.id_hogar,pw.real,pw.predicha_regla);ca=bootstrap_hogares(pw.id_hogar,pw.real,pw.predicha);del pred
    R.add('## Modelo final: resultados en prueba\n\n'
        f'El modelo final es **{w}** con {json.dumps(dec["configuraciones"][w],ensure_ascii=False)}, elegido por menor costo esperado en validación entre las familias con probabilidades. '
        f'En la prueba reservada (n = {miles(len(pw))}), la regla de costos alcanza **{1000*r.costo_regla:.0f} por mil** [IC 95 %: {1000*ci[0,0]:.0f}–{1000*ci[1,0]:.0f}], frente a **{1000*r.costo_argmax:.0f}** [{1000*ca[0,0]:.0f}–{1000*ca[1,0]:.0f}] si se asignara la clase de máxima probabilidad y **{1000*cte.costo_regla:.0f}** con la mejor clase constante (C{k+1}, elegida con las frecuencias de entrenamiento). '
        f'La F1 macro de la regla es {r.f1_macro_regla:.3f} [{ci[0,1]:.3f}–{ci[1,1]:.3f}] y la de máxima probabilidad {r.f1_macro:.3f}; la exactitud pasa de {r.exactitud:.3f} a {r.exactitud_regla:.3f}. '
        'La regla no busca maximizar aciertos: acepta más errores baratos para reducir el más caro. Los intervalos remuestrean hogares-mes de prueba con modelo, cortes y regla fijos; no incorporan el diseño muestral de la GEIH ni la incertidumbre de selección.')
    t=q.reset_index()
    R.tab(pd.DataFrame({'Modelo':t.modelo,'Seleccionado':t.ganador.astype(bool),'Costo regla por mil':mil(t.costo_regla),'F1 regla':t.f1_macro_regla,'Exactitud regla':t.exactitud_regla,'Error ordinal regla':t.error_ordinal_regla,'Costo máx. prob. por mil':mil(t.costo_argmax),'F1 máx. prob.':t.f1_macro,'AUC macro':t.auc_macro}),'Dominio nacional con ciudad: todas las familias en prueba. Las referencias son la clase mayoritaria (por máxima probabilidad) y la mejor clase constante según costo.')
    cm=pd.read_csv(pf/'confusiones.csv');cm=cm[cm.modelo==w];fig,axes=plt.subplots(1,2,figsize=(9,4.2))
    for ax,(kind,title) in zip(axes,[('costos','Regla de costos (se despliega)'),('argmax','Máxima probabilidad (referencia)')]):
        m=cm[cm.regla==kind].pivot(index='real',columns='predicha',values='n').reindex(index=range(4),columns=range(4)).to_numpy();rate=m/np.maximum(m.sum(axis=1,keepdims=True),1)
        ax.imshow(rate,vmin=0,vmax=1,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{m[i,j]:,}\n{rate[i,j]:.0%}',ha='center',va='center',fontsize=8,color='white' if rate[i,j]>.55 else 'black')
        ax.set_xticks(range(4),dc.NOMBRES);ax.set_yticks(range(4),dc.NOMBRES);ax.set_xlabel('Clase asignada');ax.set_ylabel('Clase real');ax.set_title(title,fontsize=9)
    R.picture(fig,f'Modelo final ({w}): matrices de confusión en prueba con la regla de costos y con la clase de máxima probabilidad. Porcentajes por fila.')
    R.add('### Tasas por clase frente a las demás\n\nCada clase se evalúa contra las otras tres. **TPR** (sensibilidad) es la proporción de personas de la clase que reciben esa clase; **FPR** es la proporción de personas de otras clases que la reciben por error; **VPP** es la proporción de asignados a la clase que realmente pertenecen a ella.')
    R.tab(pd.DataFrame({'Clase':dc.NOMBRES,'Prevalencia':a_.prevalencia.values,'Asignados regla':a_.predichos.values,'TPR regla':a_.TPR.values,'FPR regla':a_.FPR.values,'VPP regla':a_.VPP.values,'TPR máx. prob.':b_.TPR.values,'FPR máx. prob.':b_.FPR.values,'VPP máx. prob.':b_.VPP.values}),'Modelo final en prueba: tasas uno contra el resto con la regla desplegada y con máxima probabilidad.')
    R.add(f'Con la regla, la TPR de C1 sube de {b_.TPR["C1"]:.2f} a **{a_.TPR["C1"]:.2f}**: se identifica a más personas del grupo prioritario. El precio es una FPR de C1 de {a_.FPR["C1"]:.2f} y un VPP de {a_.VPP["C1"]:.2f}, es decir, parte de quienes reciben C1 pertenecen a otros tramos. Esa es exactamente la preferencia que expresa la matriz: una inclusión errónea en C1 cuesta menos que una exclusión.')
    cal=pd.read_csv(pf/'calibracion.csv');fig,axes=plt.subplots(2,2,figsize=(8,7))
    for ax,c in zip(axes.flat,dc.NOMBRES):
        g=cal[cal.clase==c];ax.plot([0,1],[0,1],':',color='gray');ax.scatter(g.prob_media,g.frecuencia,s=np.clip(g.n/g.n.max()*200,10,200),alpha=.7);ax.plot(g.prob_media,g.frecuencia);ax.set_title(c);ax.set_xlabel('Probabilidad media estimada');ax.set_ylabel('Frecuencia observada');ax.set_xlim(0,1);ax.set_ylim(0,1)
    R.picture(fig,'Calibración del modelo final en prueba por clase (diez intervalos de probabilidad; tamaño proporcional al número de casos). La regla de costos supone probabilidades razonablemente calibradas; desvíos respecto a la diagonal cambian sus decisiones.')
    ej=pd.read_csv(pf/'ejemplos.csv')
    R.tab(pd.DataFrame({'Caso':ej.caso,'Ciudad':ej.ciudad,'Año':ej.anio,'Edad':ej.edad,'Educación':ej.educacion,'Horas':entero(ej.horas),'Antigüedad (meses)':entero(ej.antiguedad),'Real':ej.real,'Asignada (regla)':ej.predicha,'Máx. prob.':ej.max_probabilidad}),'Ejemplos anonimizados del modelo final: dos primeros casos de prueba de cada clase asignada por la regla, incluidos errores.')
    R.tab(pd.read_csv(pf/'perfiles.csv').assign(clase=lambda x:x.clase+1),'Perfiles de los grupos asignados por la regla en prueba. El ingreso mediano es una descripción posterior, no una entrada del modelo.')
    bad=cm[(cm.regla=='costos')&(cm.real!=cm.predicha)].sort_values('n',ascending=False).iloc[0]
    R.add(f'La confusión más frecuente con la regla es **C{int(bad.real)+1} → C{int(bad.predicha)+1}**, con {miles(bad.n)} casos. Para una campaña informativa, la clase asignada sugiere qué contenido mostrar primero; nunca reemplaza el ingreso declarado ni decide acceso a empleo, crédito o beneficios. GEIH no aporta una lista de contactos: una entidad tendría que validar el procedimiento en su propia población, con datos consentidos y controles de acceso.')

def tasas_grupo(R):
    root=R.root;pf=root/'resultados_ml_final'/str(AREAS.index(FINAL))
    tg=pd.read_csv(pf/'tasas_grupo.csv');t=tg[tg.regla=='costos'].copy()
    R.add('## Tasas por grupo: TPR, FPR y VPP de cada clase frente a las demás\n\n'
        'Se calculan en la prueba del modelo final con la regla de costos desplegada, para sexo, edad (cuatro tramos), educación y ciudad. Para cada grupo y clase: n del grupo, prevalencia de la clase, número de asignados, TPR, FPR y VPP. '
        'Las tablas completas con la regla de máxima probabilidad están en el anexo. Las tasas describen desempeño; no prueban discriminación ni ausencia de ella, y las clases relativas a cada ciudad y año hacen que las prevalencias difieran entre grupos.')
    for var in ['sexo','grupo_edad','educacion','ciudad']:
        g=t[t.variable==var].copy();g['orden']=g.grupo.map({v:i for i,v in enumerate(ORDEN_GRUPOS[var])});g=g.sort_values(['orden','clase'])
        R.add('### '+VARIABLE_TXT[var])
        R.tab(pd.DataFrame({'Grupo':g.grupo.values,'Clase':g.clase.values,'n':g.n.values,'Prevalencia':g.prevalencia.values,'Asignados':g.predichos.values,'TPR':g.TPR.values,'FPR':g.FPR.values,'VPP':g.VPP.values}),f'{VARIABLE_TXT[var]}: tasas uno contra el resto del modelo final en prueba con la regla de costos.')
    filas=[(var,v) for var in ['sexo','grupo_edad','educacion','ciudad'] for v in ORDEN_GRUPOS[var]]
    fig,axes=plt.subplots(1,3,figsize=(11,7.5),sharey=True)
    for ax,tasa in zip(axes,['TPR','FPR','VPP']):
        mat=np.array([[t[(t.variable==var)&(t.grupo==v)&(t.clase==c)][tasa].squeeze() if ((t.variable==var)&(t.grupo==v)&(t.clase==c)).any() else np.nan for c in dc.NOMBRES] for var,v in filas],dtype=float)
        ax.imshow(mat,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i in range(mat.shape[0]):
            for j in range(4):ax.text(j,i,'—' if np.isnan(mat[i,j]) else f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if (not np.isnan(mat[i,j]) and mat[i,j]<.5) else 'black')
        ax.set_xticks(range(4),dc.NOMBRES);ax.set_title(tasa);ax.set_xlabel('Clase')
        for y in [1.5,5.5,9.5]:ax.axhline(y,color='white',lw=2)
    axes[0].set_yticks(range(len(filas)),[f'{VARIABLE_TXT[var]}: {v}' for var,v in filas],fontsize=7)
    R.picture(fig,'Modelo final: TPR, FPR y VPP de cada clase frente a las demás, por grupo, en prueba con la regla de costos.',height=6.2)
    b=pd.read_csv(pf/'brechas_grupo.csv');b=b[b.regla=='costos']
    R.tab(pd.DataFrame({'Variable':b.variable.map(VARIABLE_TXT).values,'Clase':b.clase.values,'Brecha TPR':b.brecha_TPR.values,'TPR menor':b.grupo_min_TPR.values,'TPR mayor':b.grupo_max_TPR.values,'Brecha FPR':b.brecha_FPR.values,'Brecha VPP':b.brecha_VPP.values}),'Brechas entre grupos (máximo − mínimo) con la regla de costos; solo grupos con al menos 30 casos en el denominador de cada tasa.')
    s=t[(t.variable=='sexo')].set_index(['grupo','clase']);c1=t[t.clase=='C1']
    def rango(var):
        g=c1[c1.variable==var].dropna(subset=['TPR']);lo=g.loc[g.TPR.idxmin()];hi=g.loc[g.TPR.idxmax()];return f'{lo.TPR:.2f} ({lo.grupo}) a {hi.TPR:.2f} ({hi.grupo})'
    R.add(f'En C1, el grupo prioritario, la TPR es {s.loc[("Hombre","C1"),"TPR"]:.2f} para hombres y {s.loc[("Mujer","C1"),"TPR"]:.2f} para mujeres, con VPP de {s.loc[("Hombre","C1"),"VPP"]:.2f} y {s.loc[("Mujer","C1"),"VPP"]:.2f}. '
        f'Por edad va de {rango("grupo_edad")}; por educación, de {rango("educacion")}; por ciudad, de {rango("ciudad")}. '
        +(lambda f:f'La FPR de C1 también varía mucho: llega a {f.FPR:.2f} en {VARIABLE_TXT[f.variable].lower()} {f.grupo}, donde la regla asigna C1 al {f.predichos/f.n:.0%} del grupo porque su probabilidad estimada de C1 casi siempre supera 0,25. Para orientar una campaña eso equivale a tratar a ese grupo casi en bloque como prioritario; para decisiones individuales sería inaceptable. ')(c1.dropna(subset=['FPR']).loc[c1.dropna(subset=['FPR']).FPR.idxmax()])+
        'Cuando la prevalencia de una clase difiere entre grupos, no es posible igualar a la vez VPP y las tasas de error salvo con un clasificador perfecto; por eso las brechas se informan sin corregirlas. '
        'Cualquier ajuste por grupo (por ejemplo, umbrales distintos) sería una decisión normativa adicional que debería tomar la entidad usuaria, no un detalle técnico del modelo.')

def ficha(R):
    R.add('## Ficha del modelo final\n\nLa ficha describe el único modelo que se despliega. Se genera desde los resultados guardados y se entrega también como modelo_final/FICHA_MODELO.md, junto con el preprocesador, el modelo serializado, los cortes por ciudad y año y la configuración con la matriz de costos. cargar_y_predecir.py aplica exactamente esta regla a perfiles nuevos y devuelve probabilidades, costos esperados y la clase asignada.')
    R.tab(pd.DataFrame(ficha_filas(R.root),columns=['Campo','Contenido']),'Ficha del modelo final.')

def comparacion_areas(R,allmetrics):
    root=R.root;pf=root/'resultados_ml_final'/str(AREAS.index(FINAL));w=loadj(pf/'decision_pre_prueba.json')['ganador']
    pn=pd.read_parquet(pf/'predicciones.parquet');pn=pn[pn.modelo==w][['id_persona','real','predicha_regla']].rename(columns={'real':'real_n','predicha_regla':'final'})
    part=pd.read_parquet(pf/'particion.parquet',columns=['id_persona','id_hogar','ciudad']);rows=[];bars=[]
    for i,a in enumerate(CIUDADES):
        p=root/'resultados_ml_final'/str(i);wa=loadj(p/'decision_pre_prueba.json')['ganador']
        pa=pd.read_parquet(p/'predicciones.parquet');pa=pa[pa.modelo==wa][['id_persona','real','predicha_regla']].rename(columns={'predicha_regla':'area'})
        m=pa.merge(pn,on='id_persona',validate='one_to_one').merge(part,on='id_persona',validate='one_to_one')
        assert len(m)==len(pa) and (m.real==m.real_n).all() and (m.ciudad==a).all()
        lo,hi=bootstrap_diferencia(m.id_hogar,m.real,m.area,m.final)
        ta=dc.tasas_ovr(m.real,m.area).set_index('clase');tf=dc.tasas_ovr(m.real,m.final).set_index('clase')
        ca=dc.costo_medio(m.real,m.area);cf=dc.costo_medio(m.real,m.final)
        rows.append({'Área':a,'Familia del área':wa,'Costo área':round(1000*ca),'Costo final':round(1000*cf),'Diferencia final − área [IC 95 %]':f'{1000*(cf-ca):+.0f} [{1000*lo:+.0f}; {1000*hi:+.0f}]',
                     'F1 área':f1_macro(m.real,m.area),'F1 final':f1_macro(m.real,m.final),'TPR C1 área':ta.loc['C1','TPR'],'TPR C1 final':tf.loc['C1','TPR']})
        bars.append((a,1000*ca,1000*cf,1000*lo,1000*hi))
    comp=pd.DataFrame(rows);mejor=sum(1 for _,ca,cf,lo,hi in bars if hi<0);peor=sum(1 for _,ca,cf,lo,hi in bars if lo>0)
    R.add('## Comparación con los modelos de las cinco áreas\n\n'
        'Los modelos por área no se despliegan: responden si un solo modelo nacional con la variable ciudad pierde desempeño frente a modelos locales. Como la partición está alineada, la comparación es **pareada**: mismas personas de prueba, mismas clases (mismos cortes por ciudad y año) y la misma regla de costos. '
        'La diferencia de costo y su intervalo se obtienen remuestreando hogares-mes de cada área.')
    R.tab(comp,'Modelo final frente al modelo propio de cada área sobre las mismas personas de prueba. Costos por mil personas; una diferencia negativa favorece al modelo final.')
    fig,ax=plt.subplots(figsize=(8,3.6));x=np.arange(len(bars))
    ax.bar(x-.2,[b[1] for b in bars],.4,label='Modelo del área',color='#9aa9c7');ax.bar(x+.2,[b[2] for b in bars],.4,label='Modelo final nacional',color='#1037B5')
    ax.set_xticks(x,[b[0] for b in bars]);ax.set_ylabel('Costo por mil personas');ax.legend(fontsize=8);ax.set_ylim(0,max(max(b[1],b[2]) for b in bars)*1.15)
    R.picture(fig,'Costo de la regla en la prueba de cada área: modelo propio del área frente al modelo final nacional, sobre las mismas personas.')
    if mejor+peor==0:concl='En ninguna área la diferencia es distinguible de cero al 95 %: un solo modelo nacional rinde como los modelos locales y simplifica el despliegue, la ficha y el mantenimiento.'
    elif peor==0:concl=f'El modelo final reduce el costo de forma distinguible de cero en {mejor} de 5 áreas y no es peor en ninguna: aprovecha la información de todo el país y, con la variable ciudad, conserva las diferencias locales.'
    else:concl=f'El modelo final es mejor en {mejor} y peor en {peor} de 5 áreas (intervalos al 95 %). Se mantiene como único modelo desplegado por coherencia de la ficha y cobertura nacional, y las áreas donde pierde se señalan como limitación.'
    R.add(concl+' Ambos enfoques comparten límites: los mismos predictores, ingresos no ponderados y cortes de años pasados.')
    import joblib
    fig,axes=plt.subplots(3,2,figsize=(9,8))
    for i,(ax,a) in enumerate(zip(axes.flat,AREAS)):
        p=root/'resultados_ml_final'/str(i);pre=joblib.load(p/'preprocesador.joblib');model=joblib.load(p/'modelo_2.joblib')['model'];names=pre.get_feature_names_out()
        imp=pd.DataFrame({'variable':[next(f for f in FEATURES_FINAL if v.split('__')[1]==f or v.split('__')[1].startswith(f+'_')) for v in names],'importancia':model.feature_importances_}).groupby('variable').importancia.sum().sort_values();ax.barh(imp.index,imp.values);ax.set_title(ETIQUETA[a]);ax.set_xlabel('Importancia por impureza');ax.tick_params(axis='y',labelsize=8)
    R.picture(fig,'Bosques aleatorios: importancia agregada por variable original en cada dominio (en el final incluye ciudad). Es descriptiva, puede favorecer variables con más cortes y no mide causalidad ni ausencia de sesgo.')
    for i,a in enumerate(CIUDADES):
        q=allmetrics[allmetrics.Dominio==a];wa=q[q.ganador.astype(bool)].modelo.item();row=comp.iloc[i]
        R.add('### '+a)
        R.tab(pd.DataFrame({'Modelo':q.modelo.values,'Seleccionado':q.ganador.astype(bool).values,'Costo regla por mil':mil(q.costo_regla),'F1 regla':q.f1_macro_regla.values,'F1 máx. prob.':q.f1_macro.values,'Exactitud máx. prob.':q.exactitud.values,'AUC macro':q.auc_macro.values}))
        R.add(f'En {a} el modelo propio elegido es **{wa}**: con la regla obtiene {row["Costo área"]} por mil en prueba, frente a {row["Costo final"]} del modelo final sobre las mismas personas (diferencia {row["Diferencia final − área [IC 95 %]"]}).')

def f1_macro(y,p):
    from sklearn.metrics import f1_score
    return f1_score(y,p,average='macro',zero_division=0)
