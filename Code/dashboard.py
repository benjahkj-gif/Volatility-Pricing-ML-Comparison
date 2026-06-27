import os
import json
import threading
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from datetime import datetime

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ─────────────────────────────────────────────
#  RUTAS DE ARCHIVOS
# ─────────────────────────────────────────────
CARPETA_DATA    = os.path.join(os.path.dirname(__file__), 'data')
CSV_DATOS       = os.path.join(CARPETA_DATA, 'opciones_msft_procesado.csv')
META_DATOS      = os.path.join(CARPETA_DATA, 'meta_descarga.json')
RESULTADOS_SVM  = os.path.join(CARPETA_DATA, 'resultados_svm.json')
RESULTADOS_XGB  = os.path.join(CARPETA_DATA, 'resultados_xgb.json')
RESULTADOS_RF   = os.path.join(CARPETA_DATA, 'resultados_rf.json')

os.makedirs(CARPETA_DATA, exist_ok=True)


# ─────────────────────────────────────────────
#  PARÁMETROS GLOBALES
# ─────────────────────────────────────────────
r = 0.045    # tasa libre de riesgo
q = 0.0075   # dividend yield MSFT


def a_lista_float(arr):
    """Convierte cualquier array numpy a lista de floats Python nativos.
    Necesario porque XGBoost devuelve float32 que json.dump no entiende."""
    return [float(x) for x in arr]


# ─────────────────────────────────────────────
#  FUNCIÓN BLACK-SCHOLES
# ─────────────────────────────────────────────
def black_scholes_call(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ─────────────────────────────────────────────
#  DESCARGA DE DATOS
# ─────────────────────────────────────────────
def descargar_datos(forzar=False):
    """
    Descarga opciones MSFT de yfinance y guarda el CSV.
    Si forzar=False y ya existe el CSV, lo reutiliza sin volver a descargar.
    """
    if not forzar and os.path.exists(CSV_DATOS):
        print('>> Usando dataset guardado, no se vuelve a descargar.')
        return True

    print('>> Descargando datos de yfinance...')
    try:
        ticker = yf.Ticker('MSFT')

        # precio spot
        hist_spot = ticker.history(period='1d')
        S0 = hist_spot['Close'].iloc[-1]
        print(f'   Precio spot MSFT: ${S0:.2f}')

        # vencimiento entre 30 y 90 días
        hoy = datetime.today()
        vencimientos = ticker.options

        opciones_validas = [
            v for v in vencimientos
            if 30 <= (datetime.strptime(v, '%Y-%m-%d') - hoy).days <= 90
        ]
        if not opciones_validas:
            opciones_validas = [vencimientos[1] if len(vencimientos) > 1 else vencimientos[0]]

        fecha_exp = opciones_validas[0]
        T = (datetime.strptime(fecha_exp, '%Y-%m-%d') - hoy).days / 365
        print(f'   Vencimiento: {fecha_exp}  (T = {T:.4f} años)')

        # cadena de opciones Call
        chain = ticker.option_chain(fecha_exp)
        calls_raw = chain.calls.copy()

        # limpieza
        df = calls_raw[[
            'strike', 'bid', 'ask', 'lastPrice',
            'impliedVolatility', 'volume', 'openInterest'
        ]].copy()

        df['midPrice'] = (df['bid'] + df['ask']) / 2

        df = df[
            (df['bid'] > 0) &
            (df['impliedVolatility'] > 0.01) &
            (df['impliedVolatility'] < 3.0) &
            (df['volume'] > 0)
        ].reset_index(drop=True)

        # variables de entrada para ML
        df['Moneyness']     = df['strike'] / S0
        df['T']             = T
        df['log_Moneyness'] = np.log(df['Moneyness'])
        df['IV_mercado']    = df['impliedVolatility']

        print(f'   Contratos útiles: {len(df)}')
        df.to_csv(CSV_DATOS, index=False)

        # metadata
        meta = {
            'S0':        round(S0, 2),
            'fecha_exp': fecha_exp,
            'T':         round(T, 4),
            'n_contratos': len(df),
            'descarga':  datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        with open(META_DATOS, 'w') as f:
            json.dump(meta, f)

        return True

    except Exception as e:
        print(f'ERROR al descargar datos: {e}')
        return False


# ─────────────────────────────────────────────
#  ENTRENAMIENTO SVR
# ─────────────────────────────────────────────
def entrenar_svm(df):
    print('>> Entrenando SVR...')

    X = df[['strike', 'Moneyness', 'log_Moneyness']]
    y = df['IV_mercado']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # grid search reducido
    c_param_range       = [1, 10, 100, 500]
    epsilon_param_range = [0.01, 0.05, 0.1]

    best_score  = -float('inf')
    best_params = {'C': 100, 'epsilon': 0.01}

    for c in c_param_range:
        for epsilon in epsilon_param_range:
            scores = cross_val_score(
                SVR(kernel='rbf', C=c, epsilon=epsilon),
                X_train_sc, y_train_sc, cv=5, n_jobs=-1
            )
            if np.mean(scores) > best_score:
                best_score  = np.mean(scores)
                best_params = {'C': c, 'epsilon': epsilon}

    print(f'   Mejores parámetros SVM: {best_params}')

    svr_model = SVR(kernel='rbf', C=best_params['C'], epsilon=best_params['epsilon'], gamma='scale')
    svr_model.fit(X_train_sc, y_train_sc)

    y_pred_sc = svr_model.predict(X_test_sc)
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    print(f'   RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}')

    # predicciones sobre todo el dataset
    X_all_sc = scaler_X.transform(df[['strike', 'Moneyness', 'log_Moneyness']])
    iv_pred_all = scaler_y.inverse_transform(
        svr_model.predict(X_all_sc).reshape(-1, 1)
    ).flatten()

    resultados = {
        'modelo':    'SVR (RBF)',
        'rmse':      rmse,
        'mae':       mae,
        'r2':        r2,
        'parametros': str(best_params),
        'features':  ['strike', 'Moneyness', 'log_Moneyness'],
        'y_test':    a_lista_float(y_test.values),
        'y_pred':    a_lista_float(y_pred),
        'iv_pred_all': a_lista_float(iv_pred_all),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(RESULTADOS_SVM, 'w') as f:
        json.dump(resultados, f)

    print('   SVM guardado.')
    return resultados


# ─────────────────────────────────────────────
#  ENTRENAMIENTO XGBOOST
# ─────────────────────────────────────────────
def entrenar_xgboost(df, S0, T):
    print('>> Entrenando XGBoost...')

    features = ['Moneyness', 'log_Moneyness', 'T']
    X = df[features].values
    y = df['IV_mercado'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'max_depth':        [3, 4, 6],
        'learning_rate':    [0.05, 0.1],
        'n_estimators':     [200, 300],
        'subsample':        [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    xgb_model = grid_search.best_estimator_
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    mae    = float(mean_absolute_error(y_test, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))

    print(f'   MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}')

    # predicciones sobre todo el dataset
    df['IV_ML'] = xgb_model.predict(df[features].values)

    sigma_constante = float(df['IV_mercado'].mean())
    df['Precio_BS_ML']    = df.apply(
        lambda row: black_scholes_call(S0, row['strike'], T, r, q, row['IV_ML']), axis=1
    )
    df['Precio_BS_Plano'] = df.apply(
        lambda row: black_scholes_call(S0, row['strike'], T, r, q, sigma_constante), axis=1
    )

    mae_bs_ml    = float(mean_absolute_error(df['midPrice'], df['Precio_BS_ML']))
    mae_bs_plano = float(mean_absolute_error(df['midPrice'], df['Precio_BS_Plano']))

    resultados = {
        'modelo':          'XGBoost',
        'rmse':            rmse,
        'mae':             mae,
        'r2':              r2,
        'parametros':      str(grid_search.best_params_),
        'features':        features,
        'y_test':          a_lista_float(y_train if len(y_train) == len(y) else y_test),
        'y_pred':          a_lista_float(y_pred),
        'iv_pred_all':     a_lista_float(df['IV_ML'].values),
        'mae_bs_ml':       mae_bs_ml,
        'mae_bs_plano':    mae_bs_plano,
        'sigma_constante': sigma_constante,
        'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(RESULTADOS_XGB, 'w') as f:
        json.dump(resultados, f)

    print('   XGBoost guardado.')
    return resultados


# ─────────────────────────────────────────────
#  ENTRENAMIENTO RANDOM FOREST
# ─────────────────────────────────────────────
def entrenar_rf(df, S0, T):
    print('>> Entrenando Random Forest...')

    features = ['Moneyness', 'log_Moneyness', 'T']
    X = df[features].values
    y = df['IV_mercado'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    df['IV_ML'] = rf.predict(X)

    y_pred_test = rf.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2   = float(r2_score(y_test, y_pred_test))

    print(f'   MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}')

    sigma_constante = float(df['IV_mercado'].median())
    df['Precio_BS_ML']    = df.apply(
        lambda row: black_scholes_call(S0, row['strike'], T, r, q, row['IV_ML']), axis=1
    )
    df['Precio_BS_Plano'] = df.apply(
        lambda row: black_scholes_call(S0, row['strike'], T, r, q, sigma_constante), axis=1
    )

    mae_bs_ml    = float(mean_absolute_error(df['midPrice'], df['Precio_BS_ML']))
    mae_bs_plano = float(mean_absolute_error(df['midPrice'], df['Precio_BS_Plano']))

    feat_imp = dict(zip(features, rf.feature_importances_.tolist()))

    resultados = {
        'modelo':           'Random Forest',
        'rmse':             rmse,
        'mae':              mae,
        'r2':               r2,
        'parametros':       'n_estimators=300, max_depth=6, min_samples_leaf=2',
        'features':         features,
        'y_test':           a_lista_float(y_test),
        'y_pred':           a_lista_float(y_pred_test),
        'iv_pred_all':      a_lista_float(df['IV_ML'].values),
        'mae_bs_ml':        mae_bs_ml,
        'mae_bs_plano':     mae_bs_plano,
        'sigma_constante':  sigma_constante,
        'feat_importancia': {k: float(v) for k, v in feat_imp.items()},
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(RESULTADOS_RF, 'w') as f:
        json.dump(resultados, f)

    print('   Random Forest guardado.')
    return resultados


# ─────────────────────────────────────────────
#  ORQUESTADOR
# ─────────────────────────────────────────────
def pipeline_completo(forzar_descarga=False):
    """
    Descarga datos (si corresponde) y entrena los 3 modelos sobre el mismo CSV.
    """
    ok = descargar_datos(forzar=forzar_descarga)
    if not ok:
        return False

    df = pd.read_csv(CSV_DATOS)

    with open(META_DATOS) as f:
        meta = json.load(f)
    S0 = meta['S0']
    T  = meta['T']

    entrenar_svm(df.copy())
    entrenar_xgboost(df.copy(), S0, T)
    entrenar_rf(df.copy(), S0, T)

    return True


# ─────────────────────────────────────────────
#  ESTADO DE ENTRENAMIENTO
# ─────────────────────────────────────────────
estado_entrenamiento = {
    'activo':  False,
    'mensaje': ''
}


def entrenar_en_background(forzar_descarga=False):
    estado_entrenamiento['activo']  = True
    estado_entrenamiento['mensaje'] = 'Descargando datos y entrenando los 3 modelos...'
    try:
        pipeline_completo(forzar_descarga=forzar_descarga)
        estado_entrenamiento['mensaje'] = 'Entrenamiento completado.'
    except Exception as e:
        estado_entrenamiento['mensaje'] = f'Error durante el entrenamiento: {e}'
    finally:
        estado_entrenamiento['activo'] = False


# ─────────────────────────────────────────────
#  LECTURA DE RESULTADOS
# ─────────────────────────────────────────────
def cargar_resultados():
    archivos = [RESULTADOS_SVM, RESULTADOS_XGB, RESULTADOS_RF]
    resultados = []
    for ruta in archivos:
        if not os.path.exists(ruta):
            return None
        try:
            with open(ruta) as f:
                contenido = f.read()
            resultados.append(json.loads(contenido))
        except (json.JSONDecodeError, OSError):
            return None
    return resultados


def cargar_meta():
    if not os.path.exists(META_DATOS):
        return None
    try:
        with open(META_DATOS) as f:
            return json.loads(f.read())
    except (json.JSONDecodeError, OSError):
        return None


def cargar_df():
    if not os.path.exists(CSV_DATOS):
        return None
    return pd.read_csv(CSV_DATOS)


# ─────────────────────────────────────────────
#  ESTILO / PALETA
# ─────────────────────────────────────────────
COLORES = {
    'SVR (RBF)':     '#00b4d8',   # cian
    'XGBoost':       '#f77f00',   # naranja
    'Random Forest': '#2ecc71',   # verde
    'mercado':       '#bdc3c7',   # gris claro
    'bs_plano':      '#e9c46a',   # dorado

    # Fondo general y superficies
    'fondo':         '#0b1120',   # azul muy oscuro (page background)
    'superficie':    '#111827',   # gris-azul oscuro (plots/cards)
    'borde':         '#374151',   # borde gris azulado
    'texto':         '#e5e7eb',   # blanco suave
}

LAYOUT_BASE = dict(
    plot_bgcolor  = COLORES['superficie'],
    paper_bgcolor = COLORES['fondo'],
    font          = dict(color=COLORES['texto'], family='monospace', size=12),
    xaxis         = dict(gridcolor=COLORES['borde'], zerolinecolor=COLORES['borde']),
    yaxis         = dict(gridcolor=COLORES['borde'], zerolinecolor=COLORES['borde']),
    legend        = dict(bgcolor='rgba(0,0,0,0)',
                         bordercolor=COLORES['borde'],
                         borderwidth=1),
    margin        = dict(t=55, b=45, l=55, r=25),
)


# ─────────────────────────────────────────────
#  GRAFICOS
# ─────────────────────────────────────────────
def grafico_sonrisa(df, resultados):
    df_plot = df.sort_values('strike')
    strikes = df_plot['strike'].values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes,
        y=df_plot['IV_mercado'].values * 100,
        mode='markers',
        name='IV mercado (real)',
        marker=dict(color=COLORES['mercado'], size=6, opacity=0.7,
                    symbol='circle-open')
    ))

    for res in resultados:
        iv_pred = np.array(res['iv_pred_all'])
        if len(iv_pred) == len(df_plot):
            fig.add_trace(go.Scatter(
                x=strikes,
                y=iv_pred * 100,
                mode='lines',
                name=res['modelo'],
                line=dict(color=COLORES[res['modelo']], width=2),
                hovertemplate='Strike: %{x}<br>IV: %{y:.2f}%<extra>' + res['modelo'] + '</extra>'
            ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Sonrisa de Volatilidad — MSFT Calls', font=dict(size=14)),
        xaxis_title='Strike (K)',
        yaxis_title='Volatilidad Implícita (%)',
        hovermode='x unified',
        height=420,
    )
    return fig


def grafico_error_por_strike(df, resultados):
    df_plot = df.sort_values('strike').copy()

    def zona(m):
        if m < 0.97:
            return 'ITM'
        elif m > 1.03:
            return 'OTM'
        return 'ATM'

    df_plot['zona'] = df_plot['Moneyness'].apply(zona)

    fig = go.Figure()

    for res in resultados:
        iv_pred = np.array(res['iv_pred_all'])
        if len(iv_pred) != len(df_plot):
            continue

        error_abs = np.abs(df_plot['IV_mercado'].values - iv_pred) * 100

        fig.add_trace(go.Scatter(
            x=df_plot['strike'].values,
            y=error_abs,
            mode='lines+markers',
            name=res['modelo'],
            line=dict(color=COLORES[res['modelo']], width=1.8),
            marker=dict(size=5),
            hovertemplate='Strike: %{x}<br>Error IV: %{y:.3f}%<extra>' + res['modelo'] + '</extra>'
        ))

    strikes_sorted = df_plot['strike'].values
    s0_aprox = strikes_sorted[len(strikes_sorted) // 2]
    fig.add_vline(x=s0_aprox * 0.97, line_dash='dot', line_color=COLORES['borde'],
                  annotation_text='ITM|ATM', annotation_font_color=COLORES['texto'])
    fig.add_vline(x=s0_aprox * 1.03, line_dash='dot', line_color=COLORES['borde'],
                  annotation_text='ATM|OTM', annotation_font_color=COLORES['texto'])

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Error Absoluto de Predicción por Strike', font=dict(size=14)),
        xaxis_title='Strike (K)',
        yaxis_title='|IV real - IV predicha| (pp)',
        hovermode='x unified',
        height=380,
    )
    return fig


def grafico_precios_bs(df, resultados, S0, T):
    df_plot = df.sort_values('strike')
    strikes = df_plot['strike'].values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strikes,
        y=df_plot['midPrice'].values,
        mode='markers',
        name='Mid-Price mercado',
        marker=dict(color=COLORES['mercado'], size=6, opacity=0.7, symbol='circle-open'),
        hovertemplate='Strike: %{x}<br>Precio: $%{y:.2f}<extra>Mercado</extra>'
    ))

    for res in resultados:
        if 'mae_bs_ml' not in res:
            continue

        iv_pred = np.array(res['iv_pred_all'])
        if len(iv_pred) != len(df_plot):
            continue

        precios_ml = [
            black_scholes_call(S0, k, T, r, q, sig)
            for k, sig in zip(strikes, iv_pred[df['strike'].argsort()])
        ]
        fig.add_trace(go.Scatter(
            x=strikes,
            y=precios_ml,
            mode='lines',
            name=f'BS + IV {res["modelo"]}',
            line=dict(color=COLORES[res['modelo']], width=2),
            hovertemplate='Strike: %{x}<br>Precio BS: $%{y:.2f}<extra>' + res['modelo'] + '</extra>'
        ))

    for res in resultados:
        if 'sigma_constante' in res:
            sigma_c = res['sigma_constante']
            precios_planos = [black_scholes_call(S0, k, T, r, q, sigma_c) for k in strikes]
            fig.add_trace(go.Scatter(
                x=strikes,
                y=precios_planos,
                mode='lines',
                name='BS plano (sigma constante)',
                line=dict(color=COLORES['bs_plano'], dash='dash', width=1.5),
            ))
            break

    if S0:
        fig.add_vline(x=S0, line_dash='dot', line_color='#555',
                      annotation_text=f'S0 = ${S0:.0f}',
                      annotation_font_color=COLORES['texto'])

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Precio Teórico Black-Scholes: IV de ML vs sigma constante', font=dict(size=14)),
        xaxis_title='Strike (K)',
        yaxis_title='Precio de la opción ($)',
        hovermode='x unified',
        height=420,
    )
    return fig


def grafico_real_vs_pred(resultados):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[r['modelo'] for r in resultados]
    )

    for i, res in enumerate(resultados, start=1):
        y_test = np.array(res['y_test'])
        y_pred = np.array(res['y_pred'])
        lim    = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

        fig.add_trace(go.Scatter(
            x=lim, y=lim,
            mode='lines',
            line=dict(color='white', dash='dash', width=1.5),
            name='Ideal',
            showlegend=(i == 1)
        ), row=1, col=i)

        fig.add_trace(go.Scatter(
            x=list(y_test), y=list(y_pred),
            mode='markers',
            marker=dict(color=COLORES[res['modelo']], opacity=0.55, size=7),
            name=res['modelo'],
            showlegend=True
        ), row=1, col=i)

        fig.update_xaxes(title_text='IV real', row=1, col=i)
        fig.update_yaxes(title_text='IV predicha', row=1, col=i)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='IV real vs IV predicha por modelo', font=dict(size=14)),
        height=400,
    )
    return fig


def grafico_metricas(resultados):
    nombres = [r['modelo'] for r in resultados]
    r2s     = [r['r2']   for r in resultados]
    maes    = [r['mae']  for r in resultados]
    rmses   = [r['rmse'] for r in resultados]
    colores = [COLORES[n] for n in nombres]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['R² (mayor es mejor)',
                        'MAE (menor es mejor)',
                        'RMSE (menor es mejor)']
    )

    fig.add_trace(go.Bar(x=nombres, y=r2s,   marker_color=colores, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=nombres, y=maes,  marker_color=colores, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=nombres, y=rmses, marker_color=colores, showlegend=False), row=1, col=3)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text='Comparación de métricas de evaluación', font=dict(size=14)),
        height=380,
    )
    return fig


# ─────────────────────────────────────────────
#  PANEL DEL MEJOR MODELO
# ─────────────────────────────────────────────
def mejor_modelo_card(resultados):
    # Ordenar por R² descendente
    ordenados = sorted(resultados, key=lambda x: x['r2'], reverse=True)
    ganador   = ordenados[0]

    filas = []
    posiciones = ['1', '2', '3']
    for i, res in enumerate(ordenados):
        es_ganador = (i == 0)
        estilo_fila = {
            'backgroundColor': '#1f2937' if es_ganador else 'transparent',
            'borderLeft': f'3px solid {COLORES[res["modelo"]]}' if es_ganador else 'none',
        }
        filas.append(html.Tr([
            html.Td(posiciones[i],
                    style={'color': COLORES[res['modelo']], 'fontWeight': 'bold',
                           'textAlign': 'center', 'fontFamily': 'monospace'}),
            html.Td(res['modelo'],
                    style={'fontWeight': 'bold' if es_ganador else 'normal',
                           'color': COLORES[res['modelo']]}),
            html.Td(f"{res['r2']:.4f}",  style={'fontFamily': 'monospace'}),
            html.Td(f"{res['mae']:.4f}", style={'fontFamily': 'monospace'}),
            html.Td(f"{res['rmse']:.4f}", style={'fontFamily': 'monospace'}),
            html.Td(res['timestamp'],    style={'color': '#9ca3af', 'fontSize': '0.8rem'}),
        ], style=estilo_fila))

    tabla = dbc.Table(
        [html.Thead(html.Tr([
            html.Th('#'),
            html.Th('Modelo'),
            html.Th('R²'),
            html.Th('MAE'),
            html.Th('RMSE'),
            html.Th('Último entrenamiento'),
        ], style={'borderBottom': f'1px solid {COLORES["borde"]}'}))] +
        [html.Tbody(filas)],
        bordered=False, hover=True, size='sm',
        style={'color': COLORES['texto'], 'marginBottom': '0'}
    )

    # Texto más descriptivo y visible
    return html.Div([
        html.Div([
            html.H4('🏆 Modelo con mejor desempeño en volatilidad implícita',
                    style={
                        'color': COLORES['texto'],
                        'fontSize': '1.0rem',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.08em',
                        'marginBottom': '6px'
                    }),
            html.H3(ganador['modelo'],
                    style={
                        'color': COLORES[ganador['modelo']],
                        'fontWeight': 'bold',
                        'fontSize': '1.4rem',
                        'marginBottom': '8px'
                    }),
        ]),
        html.Div([
            html.P(
                "Resumen de rendimiento en el conjunto de prueba:",
                style={'color': '#9ca3af', 'fontSize': '0.85rem',
                       'marginBottom': '4px'}
            ),
            html.Ul([
                html.Li(f"Coeficiente de determinación (R²): {ganador['r2']:.4f}",
                        style={'fontFamily': 'monospace', 'fontSize': '0.85rem'}),
                html.Li(f"Error medio absoluto (MAE): {ganador['mae']:.4f}",
                        style={'fontFamily': 'monospace', 'fontSize': '0.85rem'}),
                html.Li(f"Raíz del error cuadrático medio (RMSE): {ganador['rmse']:.4f}",
                        style={'fontFamily': 'monospace', 'fontSize': '0.85rem'}),
            ], style={'paddingLeft': '18px', 'marginBottom': '10px'})
        ]),
        html.P(
            f"Hiperparámetros seleccionados: {ganador['parametros']}",
            style={'fontFamily': 'monospace', 'fontSize': '0.8rem',
                   'color': '#9ca3af', 'marginBottom': '16px'}
        ),
        tabla
    ])


# ─────────────────────────────────────────────
#  APP DASH
# ─────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Dashboard — Volatilidad MSFT'
)

app.layout = html.Div(
   style={
         "backgroundColor": "#020617",          # fondo general (muy oscuro)
        "minHeight": "100vh",
        "color": "#e5e7eb",                    # texto principal claro
    },
    children=[
        dbc.Container([
            dbc.Row(dbc.Col(html.H2(
               '📊 Comparación de Modelos — Volatilidad Implícita MSFT',
               className='text-center my-4'
            ))),

            dbc.Row(dbc.Col(
                html.Div(id='banner-dataset',
                         className='alert alert-secondary text-center py-2 mb-3')
            )),

            dbc.Row([
                dbc.Col(dbc.Button(
                    '🔄 Reentrenar (mismo dataset)',
                    id='btn-reentrenar',
                    color='primary', className='w-100'
                ), width=4),
                dbc.Col(dbc.Button(
                    '🌐 Descargar datos nuevos y reentrenar',
                    id='btn-datos-nuevos',
                    color='success', className='w-100'
                ), width=4),
                dbc.Col(
                    html.Div(id='estado-entrenamiento',
                             className='text-center pt-2 text-muted')
                , width=4)
            ], className='mb-4'),

            html.Div(id='contenido-principal'),
            dcc.Interval(id='intervalo', interval=4000, n_intervals=0),
            dcc.Store(id='timestamps-previos', data={})

        ], fluid=True)
    ]
)


# ─────────────────────────────────────────────
#  CALLBACK BOTONES
# ─────────────────────────────────────────────
@app.callback(
    Output('estado-entrenamiento', 'children'),
    Input('btn-reentrenar', 'n_clicks'),
    Input('btn-datos-nuevos', 'n_clicks'),
    prevent_initial_call=True
)
def lanzar_entrenamiento(n_reentrenar, n_datos_nuevos):
    if estado_entrenamiento['activo']:
        return '⏳ Ya hay un entrenamiento corriendo...'

    ctx = dash.callback_context
    if not ctx.triggered:
        return ''

    boton_id = ctx.triggered[0]['prop_id'].split('.')[0]
    forzar   = (boton_id == 'btn-datos_nuevos')

    hilo = threading.Thread(
        target=entrenar_en_background,
        args=(forzar,),
        daemon=True
    )
    hilo.start()

    return '⏳ Entrenando... (esto puede tardar unos minutos)'


# ─────────────────────────────────────────────
#  CALLBACK PRINCIPAL
# ─────────────────────────────────────────────
@app.callback(
    Output('contenido-principal',  'children'),
    Output('banner-dataset',       'children'),
    Output('estado-entrenamiento', 'children', allow_duplicate=True),
    Output('timestamps-previos',   'data'),
    Input('intervalo',             'n_intervals'),
    State('timestamps-previos',    'data'),
    prevent_initial_call=True
)
def actualizar_dashboard(n_intervals, ts_previos):
    NO_UPDATE = dash.no_update

    ts_actual = {}
    for ruta in [RESULTADOS_SVM, RESULTADOS_XGB, RESULTADOS_RF]:
        if os.path.exists(ruta):
            ts_actual[ruta] = os.path.getmtime(ruta)

    msg_estado = NO_UPDATE
    if estado_entrenamiento['activo']:
        msg_estado = f"⏳ {estado_entrenamiento['mensaje']}"
    elif estado_entrenamiento['mensaje']:
        msg_estado = f"✅ {estado_entrenamiento['mensaje']}"

    hay_cambios = (ts_actual != ts_previos)

    if not hay_cambios and ts_previos:
        return NO_UPDATE, NO_UPDATE, msg_estado, NO_UPDATE

    meta = cargar_meta()
    if meta:
        banner = (f"📅 Datos: {meta['descarga']}  |  "
                  f"Spot S₀ = ${meta['S0']}  |  "
                  f"Vencimiento: {meta['fecha_exp']} (T={meta['T']} años)  |  "
                  f"Contratos: {meta['n_contratos']}")
    else:
        banner = 'Sin datos cargados. Usá los botones para descargar y entrenar.'

    resultados = cargar_resultados()
    df         = cargar_df()

    if resultados is None or df is None:
        contenido = dbc.Alert(
            [
                html.H5('Aún no hay modelos entrenados', className='alert-heading'),
                html.P('Hacé clic en uno de los botones de arriba para descargar los '
                       'datos y entrenar los 3 modelos. La primera vez puede tardar unos minutos.')
            ],
            color='info', className='text-center mt-4'
        )
        return contenido, banner, msg_estado, ts_actual

    S0 = meta['S0'] if meta else None
    T  = meta['T']  if meta else None

    fig_sonrisa    = grafico_sonrisa(df, resultados)
    fig_real_pred  = grafico_real_vs_pred(resultados)
    fig_metricas   = grafico_metricas(resultados)
    fig_error_strk = grafico_error_por_strike(df, resultados)
    tarjeta_ganador = mejor_modelo_card(resultados)

    contenido = html.Div([
        dbc.Row(dbc.Col(
            dbc.Card(dbc.CardBody(tarjeta_ganador), className='mb-4 shadow-sm', style={'backgroundColor': COLORES['superficie'],'border': f'1px solid {COLORES["borde"]}'})
        )),

        dbc.Row(dbc.Col(
            dcc.Graph(figure=fig_metricas)
        ), className='mb-3'),

        dbc.Row(dbc.Col(
            dcc.Graph(figure=fig_sonrisa)
        ), className='mb-3'),

        dbc.Row(dbc.Col(
            dcc.Graph(figure=fig_error_strk)
        ), className='mb-3'),

        dbc.Row(dbc.Col(
            dcc.Graph(figure=fig_real_pred)
        ), className='mb-3'),
    ])

    if S0 and T:
        fig_precios = grafico_precios_bs(df.sort_values('strike'), resultados, S0, T)
        contenido.children.append(
            dbc.Row(dbc.Col(dcc.Graph(figure=fig_precios)), className='mb-3')
        )

    return contenido, banner, msg_estado, ts_actual


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 55)
    print('  Dashboard — Volatilidad Implícita MSFT')
    print('  Abrí http://127.0.0.1:8050 en el navegador')
    print('=' * 55)
    app.run(debug=False, port=8050)