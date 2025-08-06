# --- Librerías ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --- F1: Preparación de datos ---
def preparar_datos(filepath):
    df = pd.read_excel(filepath)
    df['FechaIngreso'] = pd.to_datetime(df['FechaIngreso'])
    df = df[['FechaIngreso', 'Paciente', 'Seguro médico']]
    df = df.groupby(['FechaIngreso']).sum().sort_index()
    df = df.asfreq('W-SUN')
    df = df.fillna(method='ffill')
    return df


# --- F2: Entrenamiento ARIMA ---
def entrenar_arima(serie, columna, fecha_corte, semanas_prediccion, order, seasonal_order):
    fecha_corte_dt = pd.to_datetime(fecha_corte)
    train = serie[serie.index <= fecha_corte_dt][columna]
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    pred = model_fit.get_forecast(steps=semanas_prediccion)
    pred_mean = pred.predicted_mean
    conf_int = pred.conf_int(alpha=0.05)
    ic_inf = conf_int.iloc[:, 0]
    ic_sup = conf_int.iloc[:, 1]

    params = {
        'modelo': 'SARIMAX',
        'order': order,
        'seasonal_order': seasonal_order,
        'params': model_fit.params.to_dict(),
        'aic': model_fit.aic,
        'bic': model_fit.bic,
        'llf': model_fit.llf
    }
    return pred_mean, ic_inf, ic_sup, model_fit, params


# --- F3: Entrenamiento Prophet ---
def entrenar_prophet(serie, columna, fecha_corte, semanas_prediccion, crecimiento='linear', yearly=True, weekly=True, usar_feriados=True):
    df = serie[[columna]].reset_index().rename(columns={'FechaIngreso': 'ds', columna: 'y'})
    fecha_corte_dt = pd.to_datetime(fecha_corte)
    train = df[df['ds'] <= fecha_corte_dt]

    model = Prophet(growth=crecimiento, yearly_seasonality=yearly, weekly_seasonality=weekly)
    if usar_feriados:
        model.add_country_holidays(country_name='CO')

    model.fit(train)
    future = model.make_future_dataframe(periods=semanas_prediccion, freq='W')
    forecast = model.predict(future)

    pred = forecast.set_index('ds')['yhat']
    ic_inf = forecast.set_index('ds')['yhat_lower']
    ic_sup = forecast.set_index('ds')['yhat_upper']

    params = {
        'modelo': 'Prophet',
        'growth': crecimiento,
        'yearly_seasonality': yearly,
        'weekly_seasonality': weekly,
        'usar_feriados': usar_feriados
    }
    return pred, ic_inf, ic_sup, model, params


# --- F4: Gráfico ---
def graficar_predicciones(serie_real, fecha_corte, modelos, titulo='Predicción vs Realidad'):
    plt.figure(figsize=(12, 6))
    fecha_corte = pd.to_datetime(fecha_corte)
    entrenamiento = serie_real[serie_real.index <= fecha_corte]
    testeo = serie_real[serie_real.index > fecha_corte]

    if not entrenamiento.empty:
        plt.plot(entrenamiento.index, entrenamiento.values, label='Entrenamiento real', color='black')
    if not testeo.empty:
        plt.plot(testeo.index, testeo.values, label='Testeo real', color='gray')

    colores = ['red', 'blue', 'green', 'orange', 'purple', 'steelblue', 'teal', 'darkcyan', 'crimson', 'olive']
    for i, (nombre, datos) in enumerate(modelos.items()):
        color = colores[i % len(colores)]
        pred = datos['pred']
        ic_inf = datos.get('ic_inf')
        ic_sup = datos.get('ic_sup')
        p = datos.get('parametros', {})

        if nombre == 'SARIMAX':
            resumen = f"(order={p.get('order')}, seasonal={p.get('seasonal_order')})"
        elif nombre == 'Prophet':
            resumen = f"(growth={p.get('growth')}, yearly={p.get('yearly_seasonality')}, weekly={p.get('weekly_seasonality')}, feriados={p.get('usar_feriados')})"
        else:
            resumen = ""

        error_rmse = datos.get('error_rmse')
        error_mae = datos.get('error_mae')
        error_texto = f"RMSE: {error_rmse:.2f} | MAE: {error_mae:.2f}" if error_rmse and error_mae else ""

        label_pred = f'{nombre} {resumen} | {error_texto}'
        plt.plot(pred.index, pred.values, label=label_pred, color=color)

        if ic_inf is not None and ic_sup is not None:
            try:
                ic_inf = ic_inf.reindex(pred.index).fillna(method='ffill')
                ic_sup = ic_sup.reindex(pred.index).fillna(method='bfill')
                plt.fill_between(pred.index, ic_inf, ic_sup, color=color, alpha=0.2, label=f'IC {nombre}')
            except Exception as e:
                print(f"[Error] Intervalo {nombre}: {e}")

    plt.axvline(fecha_corte, color='gray', linestyle='--', label='Fecha de corte')
    plt.title(titulo)
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- F5: Métricas ---
def calcular_rmse(serie_real, predicciones):
    return np.sqrt(mean_squared_error(serie_real, predicciones))

def calcular_mae(serie_real, predicciones):
    return mean_absolute_error(serie_real, predicciones)


# --- F6: Gridsearch ARIMA ---
def gridsearch_arima(serie, columna, fecha_corte, semanas_prediccion, orders, seasonal_orders, metric='rmse'):
    historial = []
    mejores_resultados = {
        'score': float('inf'),
        'params': None,
        'pred': None,
        'ic_inf': None,
        'ic_sup': None,
        'modelo': None,
        'parametros': None
    }

    real = serie[columna]
    fecha_corte_dt = pd.to_datetime(fecha_corte)
    y_test = real[real.index > fecha_corte_dt].iloc[:semanas_prediccion]

    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                pred, ic_inf, ic_sup, modelo, params = entrenar_arima(
                    serie, columna, fecha_corte, semanas_prediccion, order, seasonal_order
                )
                pred = pred.reindex(y_test.index).dropna()
                y_true = y_test.reindex(pred.index).dropna()

                rmse = calcular_rmse(y_true, pred)
                mae = calcular_mae(y_true, pred)
                score = rmse if metric == 'rmse' else mae

                resultado = {
                    'score': score,
                    'params': {'order': order, 'seasonal_order': seasonal_order},
                    'pred': pred,
                    'ic_inf': ic_inf,
                    'ic_sup': ic_sup,
                    'modelo': modelo,
                    'parametros': params,
                    'error_rmse': rmse,
                    'error_mae': mae
                }

                historial.append(resultado)

                if score < mejores_resultados['score']:
                    mejores_resultados.update(resultado)

            except Exception as e:
                print(f"[Error ARIMA] order={order}, seasonal={seasonal_order}: {e}")
                continue

    return mejores_resultados, historial


# --- F7: Gridsearch Prophet ---
def gridsearch_prophet(serie, columna, fecha_corte, semanas_prediccion,
                       growth_options=['linear'], yearly_options=[True], weekly_options=[True],
                       usar_feriados_options=[True], metric='rmse'):
    historial = []
    mejores_resultados = {
        'score': float('inf'),
        'params': None,
        'pred': None,
        'ic_inf': None,
        'ic_sup': None,
        'modelo': None,
        'parametros': None
    }

    real = serie[columna]
    fecha_corte_dt = pd.to_datetime(fecha_corte)
    y_test = real[real.index > fecha_corte_dt].iloc[:semanas_prediccion]

    for growth in growth_options:
        for yearly in yearly_options:
            for weekly in weekly_options:
                for usar_feriados in usar_feriados_options:
                    try:
                        pred, ic_inf, ic_sup, modelo, params = entrenar_prophet(
                            serie, columna, fecha_corte, semanas_prediccion,
                            crecimiento=growth, yearly=yearly, weekly=weekly, usar_feriados=usar_feriados
                        )
                        pred = pred.reindex(y_test.index).dropna()
                        y_true = y_test.reindex(pred.index).dropna()

                        rmse = calcular_rmse(y_true, pred)
                        mae = calcular_mae(y_true, pred)
                        score = rmse if metric == 'rmse' else mae

                        resultado = {
                            'score': score,
                            'params': {
                                'growth': growth,
                                'yearly': yearly,
                                'weekly': weekly,
                                'usar_feriados': usar_feriados
                            },
                            'pred': pred,
                            'ic_inf': ic_inf,
                            'ic_sup': ic_sup,
                            'modelo': modelo,
                            'parametros': params,
                            'error_rmse': rmse,
                            'error_mae': mae
                        }

                        historial.append(resultado)

                        if score < mejores_resultados['score']:
                            mejores_resultados.update(resultado)

                    except Exception as e:
                        print(f"[Error Prophet] growth={growth}, yearly={yearly}, weekly={weekly}, feriados={usar_feriados}: {e}")
                        continue

    return mejores_resultados, historial
