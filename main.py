from fastapi import FastAPI, Request, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from modelo_pacientes import *  # Todo tu código va ahí
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generar")
def generar(request: Request,
            order_p: str = Form(...),
            seasonal_order_p: str = Form(...),
            growth: str = Form(...),
            yearly: str = Form(...),
            weekly: str = Form(...),
            usar_feriados: str = Form(...)):
    
    # Preprocesar inputs
    order = eval(order_p)
    seasonal_order = eval(seasonal_order_p)
    yearly = yearly == "True"
    weekly = weekly == "True"
    usar_feriados = usar_feriados == "True"

    columna = 'Seguro médico'
    fecha_corte = '2024-12-01'
    semanas_prediccion = 48

    # Preparar datos
    df_pivot = preparar_datos("Informacion Pacientes Prepagada.xlsx")

    # Entrenar y evaluar
    modelo_sarimax, dict_sarimax = entrenar_y_evaluar(
        'SARIMAX', entrenar_arima, df_pivot, columna, fecha_corte, semanas_prediccion,
        order=order, seasonal_order=seasonal_order
    )

    modelo_prophet, dict_prophet = entrenar_y_evaluar(
        'Prophet', entrenar_prophet, df_pivot, columna, fecha_corte, semanas_prediccion,
        crecimiento=growth, yearly=yearly, weekly=weekly, usar_feriados=usar_feriados
    )

    modelos = {
        modelo_sarimax: dict_sarimax,
        modelo_prophet: dict_prophet
    }

    # Generar gráfica y guardarla
    plt.switch_backend('Agg')  # Para entorno sin GUI
    graficar_predicciones(df_pivot[columna], fecha_corte, modelos, titulo="Predicción en producción")
    plt.savefig("static/grafica.png")
    plt.close()

    return templates.TemplateResponse("index.html", {"request": request, "img_url": "/static/grafica.png"})
