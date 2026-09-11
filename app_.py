"""
Centro de Monitoreo de Calidad del Aire — Alejandría
-------------------------------------------------------
Aplicación Streamlit para consulta y visualización de PM2.5
de la estación 320 (Alejandría, Antioquia).
Fuente: MARCO / CORNARE — https://marco.cornare.gov.co/geoportal/320

Ejecutar:
    streamlit run app_pm25_alejandria.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3

warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    STATSMODELS_OK = True
except ImportError:  # por si el despliegue aún no tiene statsmodels instalado
    STATSMODELS_OK = False

# ==============================================================
# CONFIGURACIÓN GENERAL
# ==============================================================
API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"
FUENTE_URL = "https://marco.cornare.gov.co/geoportal/320"

LAT_DEFECTO = 6.3800
LON_DEFECTO = -75.1400

CANDIDATOS_LAT = ["lat", "latitude", "latitud"]
CANDIDATOS_LON = ["lng", "lon", "longitude", "longitud"]

# Logo institucional — debe estar en la raíz del repo de GitHub como logo.png
LOGO_PATH = Path(__file__).with_name("logo.png")

CODIGO_ESTACION_AIRE = "320"
PERIODO_ESTACIONAL = 24  # ciclo diario para datos horarios

ESTACION_AIRE = {
    "nombre": "Estación 320",
    "ubicacion": "Alejandría, Antioquia",
    "descripcion": "Monitoreo de calidad del aire — material particulado PM2.5",
}

# Categorías de calidad del aire para PM2.5 (µg/m³, referencia ICA/EPA simplificada)
CATEGORIAS_PM25 = [
    (12.0, "Buena", "#2ecc71"),
    (35.4, "Moderada", "#f1c40f"),
    (55.4, "Dañina (grupos sensibles)", "#e67e22"),
    (150.4, "Dañina", "#e74c3c"),
    (250.4, "Muy dañina", "#9b59b6"),
    (float("inf"), "Peligrosa", "#7b241c"),
]

st.set_page_config(
    page_title="Centro de Monitoreo de Calidad del Aire — Alejandría",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================
# ESTILOS
# ==============================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #eaf3fb 0%, #ffffff 100%);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .top-header {
        background: white;
        border-radius: 18px;
        padding: 16px 24px;
        box-shadow: 0 4px 18px rgba(0, 60, 110, .10);
        border: 1px solid #cfe2f3;
        margin-bottom: 18px;
    }

    .main-title {
        color: #2f7fbf;
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0;
        margin-top: 20px;
    }

    .subtitle {
        color: #31516e;
        margin-top: 5px;
        font-size: 1rem;
    }

    .station-card {
        background: white;
        border-left: 7px solid #2f7fbf;
        border-radius: 16px;
        padding: 17px 22px;
        box-shadow: 0 4px 16px rgba(0, 60, 110, .08);
        margin: 12px 0 18px 0;
    }

    .station-name {
        color: #2f7fbf;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 15px;
    }

    .station-info {
        color: #5e6b73;
        margin-top: 4px;
    }

    .section-title {
        color: #2f7fbf;
        font-size: 1.3rem;
        font-weight: 800;
        margin: 22px 0 10px 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #cfe2f3;
        border-radius: 15px;
        padding: 13px;
        box-shadow: 0 3px 12px rgba(0, 60, 110, .06);
    }

    div[data-testid="stMetricLabel"] {
        color: #2f7fbf;
    }

    div[data-testid="stMetricValue"] {
        color: #123b68;
    }

    .info-box {
        background: #eaf3fb;
        border-radius: 12px;
        padding: 13px 17px;
        color: #1c3e5c;
        border: 1px solid #cfe2f3;
    }

    .aqi-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 168px;
        height: 30px;
        padding: 0 14px;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        font-size: .8rem;
        white-space: nowrap;
        text-align: center;
    }

    .aqi-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }

    .source-note {
        font-size: .82rem;
        color: #5e6b73;
        margin-top: 6px;
    }

    .source-note a {
        color: #2f7fbf;
        text-decoration: none;
        font-weight: 600;
    }

    .footer-section {
        background: white;
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 4px 18px rgba(0, 60, 110, .10);
        border: 1px solid #cfe2f3;
        margin-top: 26px;
    }

    .footer {
        text-align: center;
        color: #718078;
        font-size: .84rem;
        padding: 10px 0 4px;
    }

    .stButton > button {
        background-color: #6c757d;
        color: white;
        border: 1px solid #6c757d;
        border-radius: 9px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #5a6268;
        border-color: #5a6268;
        color: white;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================
# FUNCIONES COMUNES
# ==============================================================
def mostrar_logo():
    if LOGO_PATH.exists():
        st.image(LOGO_PATH, width=155)


def obtener_serie(codigo_estacion, variable, desde, hasta, calidad=1, timeout=30):
    """Consulta el endpoint .../estaciones/{codigo}/{variable} de la API MARCO/CORNARE."""
    url = f"{API_BASE_URL}/{codigo_estacion}/{variable}"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"


def obtener_todas_las_paginas(datos_json, timeout=30):
    """Sigue el campo 'next' de una respuesta paginada y acumula todos los 'values'."""
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")

    while siguiente_url:
        try:
            resp = requests.get(siguiente_url, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            break

        if resp.status_code != 200:
            break

        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")

    return registros


def detectar_coordenadas(datos_json):
    if not isinstance(datos_json, dict):
        return LAT_DEFECTO, LON_DEFECTO, False

    lat = next((datos_json[k] for k in CANDIDATOS_LAT if k in datos_json), None)
    lon = next((datos_json[k] for k in CANDIDATOS_LON if k in datos_json), None)

    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon), True
        except (TypeError, ValueError):
            pass

    return LAT_DEFECTO, LON_DEFECTO, False


# ==============================================================
# FUNCIONES — CALIDAD DEL AIRE (PM2.5)
# Tres funcionalidades de carga de datos:
#   1) cargar_pm25                  -> descarga cruda + paginación
#   2) preparar_serie_pm25          -> limpieza (huecos + outliers)
#   3) cargar_pm25_ultimos_30_dias  -> orquesta las dos anteriores
#      para el rango fijo de los últimos 30 días
# ==============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def cargar_pm25(codigo_estacion, desde, hasta, calidad=1):
    """1) Descarga la serie PM2.5 desde la API de CORNARE, siguiendo la paginación."""
    datos_json, error = obtener_serie(codigo_estacion, "PM25", desde, hasta, calidad)
    if error or not datos_json:
        return None, (LAT_DEFECTO, LON_DEFECTO, False), (error or "Sin datos disponibles")

    registros = obtener_todas_las_paginas(datos_json)
    coords = detectar_coordenadas(datos_json)
    return registros, coords, None


def preparar_serie_pm25(registros):
    """2) Limpieza de la serie: tipos, regularización horaria, interpolación de huecos
    y corrección de outliers (IQR + límite físico ≥ 0)."""
    if not registros:
        return pd.DataFrame(columns=["fecha", "valor"]), 0, 0

    df = pd.DataFrame(registros)
    col_fecha = next((c for c in ["fecha", "level_date", "date"] if c in df.columns), None)
    col_valor = next((c for c in ["muestra", "level", "valor", "value"] if c in df.columns), None)

    if col_fecha is None or col_valor is None:
        return pd.DataFrame(columns=["fecha", "valor"]), 0, 0

    df = df.rename(columns={col_fecha: "fecha", col_valor: "valor"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates(subset="fecha")

    if df.empty:
        return pd.DataFrame(columns=["fecha", "valor"]), 0, 0

    # Regulariza a frecuencia horaria e interpola huecos en el tiempo
    df_idx = df.set_index("fecha")
    rango_completo = pd.date_range(df_idx.index.min(), df_idx.index.max(), freq="h")
    df_reg = df_idx.reindex(rango_completo)
    huecos = int(df_reg["valor"].isna().sum())
    df_reg["valor"] = df_reg["valor"].interpolate(method="time").ffill().bfill()

    # Outliers: rango intercuartílico + límite físico (PM2.5 no puede ser negativo)
    q1, q3 = df_reg["valor"].quantile([.25, .75])
    iqr = q3 - q1
    lim_inf = max(q1 - 1.5 * iqr, 0)
    lim_sup = q3 + 1.5 * iqr
    es_outlier = (df_reg["valor"] < lim_inf) | (df_reg["valor"] > lim_sup)
    n_outliers = int(es_outlier.sum())

    df_reg.loc[es_outlier, "valor"] = np.nan
    df_reg["valor"] = df_reg["valor"].interpolate(method="time").ffill().bfill()

    df_limpio = df_reg.reset_index().rename(columns={"index": "fecha"})
    return df_limpio, huecos, n_outliers


def cargar_pm25_ultimos_30_dias(codigo_estacion=CODIGO_ESTACION_AIRE, calidad=1):
    """3) Combina la descarga y la limpieza para traer automáticamente
    los últimos 30 días de PM2.5 de la estación configurada."""
    hasta = pd.Timestamp.now().normalize()
    desde = hasta - pd.Timedelta(days=30)

    registros, coords, error = cargar_pm25(
        codigo_estacion,
        desde.strftime("%Y-%m-%d"),
        hasta.strftime("%Y-%m-%d"),
        calidad,
    )

    if error:
        return pd.DataFrame(), coords, 0, 0, error

    df_limpio, huecos, n_outliers = preparar_serie_pm25(registros)

    if df_limpio.empty:
        return df_limpio, coords, huecos, n_outliers, "No se encontraron datos válidos."

    return df_limpio, coords, huecos, n_outliers, None


def clasificar_pm25(valor):
    if pd.isna(valor):
        return "Sin datos", "#95a5a6"
    for limite, nombre, color in CATEGORIAS_PM25:
        if valor <= limite:
            return nombre, color
    return CATEGORIAS_PM25[-1][1], CATEGORIAS_PM25[-1][2]


def chip_categoria(valor):
    nombre, color = clasificar_pm25(valor)
    return f'<span class="aqi-chip" style="background:{color};">{nombre}</span>'


# ---- Tres funcionalidades de gráficos para los últimos 30 días ----
def grafico_serie_pm25(df):
    """1) Serie horaria completa de los últimos 30 días."""
    st.line_chart(
        df.set_index("fecha")["valor"],
        height=380,
        color="#2f7fbf",
    )


def grafico_promedio_diario_pm25(df):
    """2) Promedio diario, coloreado según la categoría de calidad del aire."""
    import altair as alt

    diario = df.set_index("fecha")["valor"].resample("D").mean().reset_index()
    diario[["categoria", "color"]] = diario["valor"].apply(
        lambda v: pd.Series(clasificar_pm25(v))
    )

    dominio = [c[1] for c in CATEGORIAS_PM25]
    rango = [c[2] for c in CATEGORIAS_PM25]

    chart = alt.Chart(diario).mark_bar().encode(
        x=alt.X("fecha:T", title="Día"),
        y=alt.Y("valor:Q", title="PM2.5 promedio (µg/m³)"),
        color=alt.Color(
            "categoria:N",
            scale=alt.Scale(domain=dominio, range=rango),
            title="Categoría",
        ),
        tooltip=[
            alt.Tooltip("fecha:T", title="Día"),
            alt.Tooltip("valor:Q", title="PM2.5 (µg/m³)", format=".1f"),
            alt.Tooltip("categoria:N", title="Categoría"),
        ],
    ).properties(height=380)

    st.altair_chart(chart, use_container_width=True)


def grafico_patron_horario_pm25(df):
    """3) Patrón horario promedio (ciclo diario típico de las últimas 30 días)."""
    import altair as alt

    horario = df.copy()
    horario["hora"] = horario["fecha"].dt.hour
    patron = horario.groupby("hora", as_index=False)["valor"].mean()

    chart = alt.Chart(patron).mark_area(
        line={"color": "#2f7fbf"},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="#ffffff", offset=0),
                alt.GradientStop(color="#9fc9ec", offset=1),
            ],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        x=alt.X("hora:O", title="Hora del día"),
        y=alt.Y("valor:Q", title="PM2.5 promedio (µg/m³)"),
        tooltip=[
            alt.Tooltip("hora:O", title="Hora"),
            alt.Tooltip("valor:Q", title="PM2.5 (µg/m³)", format=".1f"),
        ],
    ).properties(height=330)

    st.altair_chart(chart, use_container_width=True)


def pronosticar_pm25_dia_siguiente(df):
    """Pronóstico de las próximas 24 horas (día siguiente).
    Usa Holt-Winters (estacionalidad diaria) si hay suficiente historia,
    con respaldo en SES o promedio simple si los datos son escasos."""
    serie = df.set_index("fecha")["valor"].asfreq("h")
    serie = serie.interpolate(method="time").ffill().bfill()

    if serie.empty:
        return pd.Series(dtype=float), "Sin datos suficientes", None

    if not STATSMODELS_OK or len(serie) < 8:
        ultimo_promedio = serie.tail(24).mean()
        fechas_futuras = pd.date_range(serie.index[-1], periods=25, freq="h")[1:]
        pronostico = pd.Series([ultimo_promedio] * 24, index=fechas_futuras)
        return pronostico, "Promedio simple (datos insuficientes)", None

    modelo_nombre = "Holt-Winters (estacional diario)"
    try:
        if len(serie) >= PERIODO_ESTACIONAL * 2:
            modelo = ExponentialSmoothing(
                serie, trend="add", seasonal="add",
                seasonal_periods=PERIODO_ESTACIONAL,
                initialization_method="estimated",
            ).fit()
        else:
            modelo_nombre = "Suavizado exponencial simple (SES)"
            modelo = SimpleExpSmoothing(serie, initialization_method="estimated").fit()
        pronostico_valores = np.asarray(modelo.forecast(24))
    except Exception:
        modelo_nombre = "Promedio simple (respaldo)"
        pronostico_valores = np.full(24, serie.tail(24).mean())

    pronostico_valores = np.clip(pronostico_valores, 0, None)
    fechas_futuras = pd.date_range(serie.index[-1], periods=25, freq="h")[1:]
    pronostico = pd.Series(pronostico_valores, index=fechas_futuras)

    # Backtest rápido: reentrena sin las últimas 24h reales para estimar el error
    error_mae = None
    if len(serie) > PERIODO_ESTACIONAL * 2 + 24:
        try:
            train_bt, test_bt = serie.iloc[:-24], serie.iloc[-24:]
            if modelo_nombre.startswith("Holt-Winters"):
                m_bt = ExponentialSmoothing(
                    train_bt, trend="add", seasonal="add",
                    seasonal_periods=PERIODO_ESTACIONAL,
                    initialization_method="estimated",
                ).fit()
            else:
                m_bt = SimpleExpSmoothing(train_bt, initialization_method="estimated").fit()
            pred_bt = np.clip(np.asarray(m_bt.forecast(24)), 0, None)
            error_mae = float(np.mean(np.abs(test_bt.values - pred_bt)))
        except Exception:
            error_mae = None

    return pronostico, modelo_nombre, error_mae


# ==============================================================
# ENCABEZADO
# ==============================================================
header_left, header_right = st.columns([5, 1])

with header_left:
    st.markdown(
        '<div class="main-title">CENTRO DE MONITOREO DE CALIDAD DEL AIRE — ALEJANDRÍA</div>'
        '<div class="subtitle">Sistema de consulta, visualización y análisis '
        'de información de calidad del aire (PM2.5)</div>',
        unsafe_allow_html=True,
    )

with header_right:
    mostrar_logo()


# ==============================================================
# CALIDAD DEL AIRE (PM2.5) — ESTACIÓN 320
# ==============================================================
st.markdown(
    '<div class="section-title">🌫️ Calidad del aire — PM2.5 (Estación 320)</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="station-card">
        <div class="station-name">📍 {ESTACION_AIRE["nombre"]}</div>
        <div class="station-info">
            <b>Código:</b> {CODIGO_ESTACION_AIRE}
            &nbsp; · &nbsp;
            <b>Ubicación:</b> {ESTACION_AIRE["ubicacion"]}
            &nbsp; · &nbsp;
            <b>Variable:</b> PM2.5 (µg/m³)
        </div>
        <div class="station-info">{ESTACION_AIRE["descripcion"]}</div>
        <div class="source-note">Fuente de datos: MARCO / CORNARE —
            <a href="{FUENTE_URL}" target="_blank">{FUENTE_URL}</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_aire1, col_aire2 = st.columns([1, 5])
with col_aire1:
    calidad_aire = st.selectbox(
        "Calidad de datos",
        [1, 0],
        index=0,
        format_func=lambda x: "Validados" if x == 1 else "Todos",
        key="calidad_aire",
    )
with col_aire2:
    st.write("")
    actualizar_aire = st.button("🔄 Actualizar calidad del aire")

if actualizar_aire:
    cargar_pm25.clear()

with st.spinner("Cargando los últimos 30 días de PM2.5..."):
    df_pm25, coords_aire, huecos_aire, n_outliers_aire, error_aire = cargar_pm25_ultimos_30_dias(
        CODIGO_ESTACION_AIRE, calidad_aire
    )

lat_aire, lon_aire, coords_aire_reales = coords_aire

if error_aire:
    st.error(f"❌ No fue posible consultar la estación de aire: {error_aire}")

else:
    ultimo_valor = df_pm25.iloc[-1]["valor"]
    categoria_actual, color_actual = clasificar_pm25(ultimo_valor)

    pronostico, modelo_nombre, error_mae = pronosticar_pm25_dia_siguiente(df_pm25)
    promedio_pronostico = pronostico.mean() if not pronostico.empty else np.nan
    categoria_pronostico, _ = clasificar_pm25(promedio_pronostico)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PM2.5 promedio (30d)", f"{df_pm25['valor'].mean():.1f} µg/m³")
    m2.metric("PM2.5 máximo (30d)", f"{df_pm25['valor'].max():.1f} µg/m³")
    m3.metric("Última lectura", f"{ultimo_valor:.1f} µg/m³")
    m4.metric("PM2.5 pronosticado (mañana)", f"{promedio_pronostico:.1f} µg/m³" if not np.isnan(promedio_pronostico) else "—")
    m5.metric("Calidad de la serie", f"{max(0, 100 - int(100 * huecos_aire / max(len(df_pm25), 1)))} / 100")

    st.markdown(
        f'Categoría actual: {chip_categoria(ultimo_valor)}'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;Categoría pronosticada mañana: {chip_categoria(promedio_pronostico)}',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="aqi-legend">' + "".join(
            f'<span class="aqi-chip" style="background:{color};">{nombre}</span>'
            for _, nombre, color in CATEGORIAS_PM25
        ) + '</div>',
        unsafe_allow_html=True,
    )

    tab_serie_a, tab_prom_a, tab_patron_a, tab_pron_a, tab_datos_a = st.tabs(
        ["📈 Serie 30 días", "📊 Promedio diario", "🕐 Patrón horario", "🔮 Pronóstico mañana", "📋 Datos"]
    )

    with tab_serie_a:
        st.markdown("#### PM2.5 — últimos 30 días (dato horario)")
        grafico_serie_pm25(df_pm25)
        st.markdown(
            f"""
            <div class="info-box">
            <b>Huecos detectados:</b> {huecos_aire} &nbsp;·&nbsp;
            <b>Outliers corregidos:</b> {n_outliers_aire}<br>
            Los huecos se interpolan en el tiempo y los outliers se corrigen con el
            criterio de rango intercuartílico (IQR) + límite físico (PM2.5 ≥ 0).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_prom_a:
        st.markdown("#### Promedio diario por categoría de calidad del aire")
        grafico_promedio_diario_pm25(df_pm25)

    with tab_patron_a:
        st.markdown("#### Patrón horario típico (ciclo diario, últimos 30 días)")
        grafico_patron_horario_pm25(df_pm25)

    with tab_pron_a:
        st.markdown("#### Pronóstico de PM2.5 para las próximas 24 horas")
        st.caption(f"Modelo utilizado: **{modelo_nombre}**" + (
            f" · Error medio absoluto estimado (backtest 24h): **{error_mae:.1f} µg/m³**"
            if error_mae is not None else ""
        ))

        historico_reciente = df_pm25.set_index("fecha")["valor"].tail(7 * 24)
        comparacion = pd.concat(
            [
                historico_reciente.rename("Histórico (últimos 7 días)"),
                pronostico.rename("Pronóstico próximas 24h"),
            ],
            axis=1,
        )
        st.line_chart(comparacion, height=400)

        st.markdown(
            f"""
            <div class="info-box">
            El pronóstico se genera reentrenando el modelo sobre la serie limpia
            de los últimos 30 días y proyectando las próximas 24 horas (día siguiente).
            PM2.5 promedio pronosticado: <b>{promedio_pronostico:.1f} µg/m³</b>
            ({chip_categoria(promedio_pronostico)}).
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_datos_a:
        st.markdown("#### Registros PM2.5 (limpios, últimos 30 días)")
        st.dataframe(df_pm25, use_container_width=True, hide_index=True)
        csv_aire = df_pm25.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Descargar datos PM2.5 CSV",
            csv_aire,
            file_name=f"pm25_estacion_{CODIGO_ESTACION_AIRE}_ultimos_30_dias.csv",
            mime="text/csv",
        )


# ==============================================================
# PIE DE PÁGINA CON MAPA
# ==============================================================
st.markdown('<div class="footer-section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📍 Ubicación de la estación</div>', unsafe_allow_html=True)

foot_map, foot_text = st.columns([2, 1])

with foot_map:
    st.map(
        pd.DataFrame({"lat": [lat_aire], "lon": [lon_aire]}),
        zoom=11,
    )

with foot_text:
    mostrar_logo()
    st.markdown(
        f"""
        <div class="station-info">
            <b>Estación 320</b> — Calidad del aire PM2.5<br>
            Alejandría, Antioquia<br><br>
            Fuente de datos: MARCO / CORNARE<br>
            <a href="{FUENTE_URL}" target="_blank">{FUENTE_URL}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        <b>Centro de Monitoreo de Calidad del Aire — Alejandría</b><br>
        Monitoreo y visualización de información de calidad del aire (PM2.5)<br>
        Información consultada desde MARCO / CORNARE
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
