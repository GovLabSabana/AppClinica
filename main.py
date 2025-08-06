from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
import pandas as pd
import os
from modelo_pacientes import preparar_datos, entrenar_arima, entrenar_prophet, calcular_rmse, calcular_mae, graficar_predicciones

os.makedirs("static", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generar")
async def generar(request: Request,
                  archivo: UploadFile = File(...),
                  order: str = Form(...),
                  seasonal_order: str = Form(...),
                  growth: str = Form(...),
                  yearly: str = Form(...),
                  weekly: str = Form(...),
                  usar_feriados: str = Form(...)):

    # Guardar archivo subido
    contenido = await archivo.read()
    filepath = f"temp_{archivo.filename}"
    with open(filepath, "wb") as f:
        f.write(contenido)

    # Convertir parámetros
    order = eval(order)
    seasonal_order = eval(seasonal_order)
    yearly = yearly == "True"
    weekly = weekly == "True"
    usar_feriados = usar_feriados == "True"

    # Variables fijas
    columna = 'Seguro médico'
    fecha_corte = '2024-12-01'
    semanas_prediccion = 48

    # Preparar datos
    df_pivot = preparar_datos(filepath)
    fecha_corte_dt = pd.to_datetime(fecha_corte)
    y_test = df_pivot[columna][df_pivot.index > fecha_corte_dt].iloc[:semanas_prediccion]

    # Entrenar SARIMAX
    pred_arima, ic_arima_inf, ic_arima_sup, modelo_arima, params_arima = entrenar_arima(
        df_pivot, columna, fecha_corte, semanas_prediccion,
        order=order, seasonal_order=seasonal_order
    )
    pred_arima_alineado = pred_arima.reindex(y_test.index).dropna()
    y_test_arima = y_test.reindex(pred_arima_alineado.index).dropna()
    error_arima_rmse = calcular_rmse(y_test_arima, pred_arima_alineado)
    error_arima_mae = calcular_mae(y_test_arima, pred_arima_alineado)

    # Entrenar Prophet
    pred_prophet, ic_prophet_inf, ic_prophet_sup, modelo_prophet, params_prophet = entrenar_prophet(
        df_pivot, columna, fecha_corte, semanas_prediccion,
        crecimiento=growth, yearly=yearly, weekly=weekly, usar_feriados=usar_feriados
    )
    pred_prophet_alineado = pred_prophet.reindex(y_test.index).dropna()
    y_test_prophet = y_test.reindex(pred_prophet_alineado.index).dropna()
    error_prophet_rmse = calcular_rmse(y_test_prophet, pred_prophet_alineado)
    error_prophet_mae = calcular_mae(y_test_prophet, pred_prophet_alineado)

    # Diccionario para graficar
    modelos = {
        'SARIMAX': {
            'pred': pred_arima,
            'ic_inf': ic_arima_inf,
            'ic_sup': ic_arima_sup,
            'parametros': params_arima,
            'error_rmse': error_arima_rmse,
            'error_mae': error_arima_mae
        },
        'Prophet': {
            'pred': pred_prophet,
            'ic_inf': ic_prophet_inf,
            'ic_sup': ic_prophet_sup,
            'parametros': params_prophet,
            'error_rmse': error_prophet_rmse,
            'error_mae': error_prophet_mae
        }
    }

    # Graficar
    plt.switch_backend('Agg')
    graficar_predicciones(df_pivot[columna], fecha_corte, modelos)
    plt.savefig("static/grafica.png")
    plt.close()

    # Borrar archivo temporal
    os.remove(filepath)

    return templates.TemplateResponse("index.html", {"request": request, "img_url": "/static/grafica.png"})
teResponse("index.html", {"request": request, "img_url": "/static/grafica.png"})


