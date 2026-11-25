"""Series muestrales GEIH: fuentes DANE, sin imputación de ingresos ni periodos."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import dane_data as dd
from dane_data._io import read_table
from dane_data.geih.loader import classify_file

AREAS = {5: 'Medellín AM', 8: 'Barranquilla AM', 11: 'Bogotá',
         68: 'Bucaramanga AM', 76: 'Cali AM'}
INDICADORES = {'p10': 'P10', 'mediana': 'Mediana / P50', 'p90': 'P90', 'promedio': 'Promedio'}
MODULOS = ['caracteristicas', 'ocupados']
VARIABLES = {'DIRECTORIO', 'SECUENCIA_P', 'HOGAR', 'ORDEN', 'AREA', 'CLASE',
             'P6040', 'P6430', 'P6450', 'P6460', 'P6800', 'P6426', 'INGLABO'}

def leer_union(archivos):
    """Lee las columnas originales y valida la unión; no elimina duplicados."""
    frames = {}
    for modulo in MODULOS:
        if not archivos.get(modulo):
            raise ValueError(f'Falta módulo {modulo}')
        frames[modulo] = pd.concat(
            [read_table(Path(f), columns=VARIABLES.__contains__) for f in archivos[modulo]],
            ignore_index=True)
    o, p = frames['ocupados'], frames['caracteristicas']
    keys = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']
    if 'HOGAR' in o and 'HOGAR' in p:
        keys.insert(2, 'HOGAR')
    for d in (o, p):
        if not set(keys).issubset(d):
            raise ValueError('Faltan llaves personales; no se puede validar la unión')
        if d[keys].isna().any().any() or d.duplicated(keys).any():
            raise ValueError('Llaves personales vacías o duplicadas')
    # Ocupados es la fuente del ingreso, contrato y área; personas aporta edad.
    extra = [c for c in p if c not in o]
    x = o.merge(p[keys + extra], on=keys, how='left', validate='one_to_one', indicator=True)
    if not x['_merge'].eq('both').all():
        raise ValueError('Hay ocupados sin correspondencia en características')
    return x.drop(columns='_merge')

def seleccionar(df, edad_minima=18, ingreso_minimo=100_000, horas_maximas=100,
                filtrar_antiguedad=True):
    oblig = ['INGLABO', 'P6040', 'P6430', 'P6450', 'P6460', 'P6800']
    if filtrar_antiguedad:
        oblig.append('P6426')
    faltan = [c for c in oblig if c not in df or not df[c].notna().any()]
    if faltan:
        raise ValueError(f'Variables faltantes o totalmente vacías: {faltan}')
    d = df.copy()
    for c in set(oblig + ['AREA', 'CLASE']) & set(d):
        # read_table ya resuelve separadores DANE. No quitar puntos a floats.
        d[c] = pd.to_numeric(d[c], errors='coerce')
    mask = (d.P6040.ge(edad_minima) & d.P6430.isin([1, 2]) & d.P6450.eq(2)
            & d.P6460.isin([1, 2]) & d.P6800.gt(0) & d.P6800.le(horas_maximas)
            & d.INGLABO.ge(ingreso_minimo) & np.isfinite(d.INGLABO))
    if filtrar_antiguedad:
        a = d.P6426.mask(d.P6426.isin([998, 999]) | d.P6426.lt(0))
        mask &= ~a.div(12).gt(d.P6040 - 14).fillna(False)
    return d.loc[mask].copy()

def estadisticas(x):
    y = np.asarray(x, dtype=float)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError('Muestra vacía o ingresos no finitos')
    q = np.quantile(y, [.1, .5, .9], method='linear')
    return dict(n=len(y), p10=q[0], mediana=q[1], p90=q[2], promedio=y.mean())

def fuentes(mes, ciudades):
    unificados = all(mes.archivos.get(m) and all(classify_file(f.name)[1] is None
                     for f in mes.archivos[m]) for m in MODULOS)
    if ciudades:
        # Históricamente Área puede tener variables ausentes en Cabecera.
        areas = {m: [] for m in MODULOS}
        for f in mes.carpeta.rglob('*'):
            if f.suffix.lower() in {'.csv', '.txt', '.dta'}:
                m, dominio = classify_file(f.name)
                if m in areas and dominio == 'area':
                    areas[m].append(f)
        if all(areas.values()):
            return areas, 'Archivos Área'
        if mes.anio < 2022 and not unificados:
            raise ValueError('Sin archivos Área verificables: no se usan departamentos ni Cabecera como sustituto')
    if mes.partes_faltantes(MODULOS):
        raise ValueError(f'Dominios incompletos: {mes.partes_faltantes(MODULOS)}')
    if not ciudades and mes.anio < 2022 and not unificados:
        for m in MODULOS:
            dominios = {classify_file(f.name)[1] for f in mes.archivos.get(m, [])}
            if not {'cabecera', 'resto'}.issubset(dominios):
                raise ValueError('Nacional histórico requiere Cabecera y Resto')
    return mes.archivos, 'Nacional original'

def ejecutar(anios, raiz, salida, descargar=True, **filtros):
    anios = sorted(set(anios))
    raiz, salida = Path(raiz).resolve(), Path(salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    mensual, anual, audit, downloads = [], [], [], []
    for anio in anios:
        print(f'Procesando {anio}', flush=True)
        if descargar:
            try:
                errores = dd.geih.download(anio, root=raiz, modules=MODULOS, delete_zips=False)
                downloads.extend({'anio': anio, 'error': str(e)} for e in errores)
            except Exception as e:
                downloads.append({'anio': anio, 'error': repr(e)})
        meses = {m.mes: m for m in dd.geih.find_months(anio, root=raiz)}
        muestras = {nombre: [] for nombre in [*AREAS.values(), 'Nacional']}
        for mes in range(1, 13):
            for ciudades in (True, False):
                nombres = list(AREAS.values()) if ciudades else ['Nacional']
                try:
                    if mes not in meses:
                        raise ValueError('Mes no disponible en disco')
                    archivos, origen = fuentes(meses[mes], ciudades)
                    raw = leer_union(archivos)
                    if not ciudades and origen == 'Nacional original':
                        if 'CLASE' not in raw or not {1, 2}.issubset(set(pd.to_numeric(raw.CLASE, errors='coerce').dropna())):
                            raise ValueError('No se verifican cabeceras y resto en CLASE del archivo nacional')
                    if ciudades and 'AREA' in raw and 'CLASE' not in raw and origen == 'Archivos Área':
                        # Recuperar el dominio observado del mismo mes/persona.
                        # No inferir CLASE a partir del nombre del archivo.
                        nacional = leer_union(meses[mes].archivos)
                        if 'CLASE' not in nacional:
                            raise ValueError('CLASE ausente también en la fuente nacional')
                        keys = ['DIRECTORIO', 'SECUENCIA_P', 'ORDEN']
                        if 'HOGAR' in raw and 'HOGAR' in nacional:
                            keys.insert(2, 'HOGAR')
                        raw = raw.merge(nacional[keys + ['CLASE']], on=keys,
                                        how='left', validate='one_to_one', indicator=True)
                        if not raw['_merge'].eq('both').all() or raw.CLASE.isna().any():
                            raise ValueError('No se pudo verificar CLASE para todas las personas de Área')
                        raw = raw.drop(columns='_merge')
                        origen += '; CLASE enlazada con fuente nacional'
                    d = seleccionar(raw, **filtros)
                    if ciudades and not {'AREA', 'CLASE'}.issubset(d):
                        raise ValueError('Falta AREA o CLASE: no se asignan ciudades')
                    for nombre in nombres:
                        if ciudades:
                            codigo = next(k for k, v in AREAS.items() if v == nombre)
                            grupo = d.loc[d.AREA.eq(codigo) & d.CLASE.eq(1)]
                        else:
                            grupo = d
                        if grupo.empty:
                            audit.append(dict(anio=anio, mes=mes, ciudad=nombre, estado='SIN_MUESTRA', detalle=origen))
                            continue
                        mensual.append(dict(anio=anio, mes=mes, ciudad=nombre,
                                            **estadisticas(grupo.INGLABO)))
                        muestras[nombre].append(grupo.INGLABO.to_numpy(dtype=float))
                        audit.append(dict(anio=anio, mes=mes, ciudad=nombre, estado='OK',
                                          detalle=origen, n=len(grupo), leidos=len(raw),
                                          archivos=' | '.join(str(f) for m in MODULOS for f in archivos[m])))
                except Exception as e:
                    audit.extend(dict(anio=anio, mes=mes, ciudad=n, estado='ERROR', detalle=str(e)) for n in nombres)
        for nombre, arrays in muestras.items():
            # Distribución de personas-mes del año, nunca promedio de medianas.
            resumen = estadisticas(np.concatenate(arrays)) if arrays else dict(n=0)
            completo = len(arrays) == 12
            if not completo:
                resumen.update({k: np.nan for k in INDICADORES})
            anual.append(dict(anio=anio, ciudad=nombre, meses_validos=len(arrays),
                              estado='completo' if completo else 'incompleto', **resumen))
        pd.DataFrame(audit).to_csv(salida/'auditoria.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(downloads, columns=['anio', 'error']).to_csv(salida/'descargas.csv', index=False)
    cols = ['anio', 'mes', 'ciudad', 'n', *INDICADORES]
    m = pd.DataFrame(mensual, columns=cols)
    grid = pd.MultiIndex.from_product([anios, range(1, 13), [*AREAS.values(), 'Nacional']], names=['anio', 'mes', 'ciudad'])
    m = m.set_index(['anio', 'mes', 'ciudad']).reindex(grid).reset_index()
    m['fecha'] = pd.to_datetime(dict(year=m.anio, month=m.mes, day=1))
    a = pd.DataFrame(anual)
    m.to_csv(salida/'mensual.csv', index=False, encoding='utf-8-sig')
    a.to_csv(salida/'anual.csv', index=False, encoding='utf-8-sig')
    (salida/'configuracion.json').write_text(json.dumps(dict(anios=anios, filtros=filtros,
        ponderado=False, unidad='COP corrientes', dane_data='0.5.0'), indent=2), encoding='utf-8')
    return m, a, pd.DataFrame(audit)

def graficar(mensual, anual, salida, mostrar=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    salida = Path(salida)
    figuras = []
    for frecuencia, datos, eje in [('mensual', mensual, 'fecha'), ('anual', anual, 'anio')]:
        fig = make_subplots(rows=5, cols=1, subplot_titles=list(AREAS.values()), vertical_spacing=.045)
        for fila, ciudad in enumerate(AREAS.values(), 1):
            d = datos.loc[datos.ciudad.eq(ciudad)].sort_values(eje)
            # Añadir años no seleccionados como huecos, también en ejecuciones parciales.
            indice = (pd.date_range(f'{datos.anio.min()}-01-01', f'{datos.anio.max()}-12-01', freq='MS')
                      if frecuencia == 'mensual' else range(datos.anio.min(), datos.anio.max()+1))
            d = d.set_index(eje).reindex(indice)
            for (col, etiqueta), color in zip(INDICADORES.items(), ['#00897b', '#1565c0', '#ef6c00', '#8e24aa']):
                fig.add_trace(go.Scatter(x=d.index, y=d[col]/1e6, name=etiqueta,
                    legendgroup=col, showlegend=fila == 1, connectgaps=False,
                    mode='lines+markers', line_color=color), row=fila, col=1)
            if datos.anio.min() <= 2022 <= datos.anio.max():
                fig.add_vline(x='2022-01-01' if frecuencia == 'mensual' else 2022,
                              line_dash='dash', line_color='gray', row=fila, col=1)
            fig.update_yaxes(title_text='Millones COP', row=fila, col=1)
        fig.update_layout(height=1500, template='plotly_white',
            title=f'GEIH · {frecuencia} · ingreso laboral mensual · COP corrientes · sin ponderar',
            legend=dict(orientation='h'))
        fig.write_html(salida/f'cinco_ciudades_{frecuencia}.html', include_plotlyjs=True)
        if mostrar:
            fig.show()
        figuras.append(fig)
    return figuras
