```python
"""
App de Streamlit — Predictor de PM2.5 / PM10 (API de CORNARE / MARCO)

Carga uno o dos modelos .pkl y genera pronósticos hacia adelante.
Cada modelo tendrá su propia gráfica.
"""

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predictor PM2.5 / PM10 — CORNARE",
    page_icon="🌫️",
    layout="wide",
)

TIPOS_DIRECTOS = {"ses", "holt_winters", "arima", "sarima"}

NOMBRES_TIPO = {
    "media_movil": "Media móvil",
    "ses": "SES",
    "holt_winters": "Holt-Winters",
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "ventana_deslizante": "Ventana deslizante",
}


def pronosticar(paquete, pasos):
    tipo = paquete["tipo"]

    if tipo in TIPOS_DIRECTOS:
        return np.asarray(paquete["modelo"].forecast(pasos))

    if tipo == "ventana_deslizante":
        modelo = paquete["modelo"]
        ventana = paquete["tamano_ventana"]
        historial = list(paquete["ultimos_valores"])
        predicciones = []

        for _ in range(pasos):
            entrada = np.array(historial[-ventana:]).reshape(1, -1)
            siguiente = modelo.predict(entrada)[0]
            predicciones.append(siguiente)
            historial.append(siguiente)

        return np.array(predicciones)

    if tipo == "media_movil":
        ventana = paquete["ventana"]
        historial = list(paquete["ultimos_valores"])
        predicciones = []

        for _ in range(pasos):
            siguiente = np.mean(historial[-ventana:])
            predicciones.append(siguiente)
            historial.append(siguiente)

        return np.array(predicciones)

    raise ValueError(
        f"Tipo de modelo desconocido: '{tipo}'. "
        "¿Es un .pkl generado por este notebook?"
    )


def extraer_historico(paquete):
    historico = paquete.get("historico")

    if not historico:
        return None

    fechas = pd.to_datetime(historico["fechas"])

    return pd.Series(
        historico["valores"],
        index=fechas
    )


def etiqueta_modelo(paquete, nombre_archivo):
    meta = paquete.get("metadata", {})

    nombre = meta.get("nombre_modelo") or NOMBRES_TIPO.get(
        paquete.get("tipo"),
        paquete.get("tipo")
    )

    return nombre or nombre_archivo


def cargar_paquete(archivo_subido):
    return joblib.load(
        io.BytesIO(archivo_subido.getvalue())
    )


st.title("🌫️ Predictor de calidad del aire — CORNARE (MARCO)")

st.caption(
    "Carga uno o dos modelos .pkl y genera predicciones de PM2.5 / PM10."
)


# --------------------------------------------------------------------------
# CARGA DE MODELOS
# --------------------------------------------------------------------------

st.sidebar.header("1. Carga tus modelos")

archivos_subidos = st.sidebar.file_uploader(
    "Sube uno o dos archivos .pkl",
    type=["pkl"],
    accept_multiple_files=True,
    help="Puedes subir hasta dos modelos."
)


if not archivos_subidos:
    st.info(
        "👈 Sube al menos un archivo .pkl en la barra lateral para comenzar."
    )

    st.markdown(
        """
        **¿No tienes los archivos a mano?**

        Corre la sección 11B del notebook *Módulo puente —
        Pronóstico de PM2.5 y PM10* para generar los modelos.
        """
    )

    st.stop()


if len(archivos_subidos) > 2:
    st.sidebar.warning(
        "Solo se utilizarán los dos primeros archivos."
    )

    archivos_subidos = archivos_subidos[:2]


# --------------------------------------------------------------------------
# CARGAR PAQUETES
# --------------------------------------------------------------------------

paquetes = []

for archivo in archivos_subidos:

    try:

        paquete = cargar_paquete(archivo)

        paquetes.append(
            (archivo.name, paquete)
        )

    except Exception as e:

        st.sidebar.error(
            f"No pude cargar {archivo.name}: {e}"
        )


if not paquetes:
    st.stop()


# --------------------------------------------------------------------------
# INFORMACIÓN DE LOS MODELOS
# --------------------------------------------------------------------------

st.sidebar.header("2. Modelos cargados")

for nombre_archivo, paquete in paquetes:

    meta = paquete.get("metadata", {})

    with st.sidebar.expander(
        f"📦 {etiqueta_modelo(paquete, nombre_archivo)}",
        expanded=True
    ):

        st.write(
            f"**Archivo:** {nombre_archivo}"
        )

        st.write(
            f"**Tipo:** {paquete.get('tipo', '—')}"
        )

        st.write(
            f"**Variable:** {meta.get('variable', '—')}"
        )

        st.write(
            f"**Estación:** {meta.get('codigo_estacion', '—')}"
        )

        rmse = meta.get("rmse_en_test")

        if rmse is not None:
            st.write(
                f"**RMSE en test:** {rmse:.3f}"
            )
        else:
            st.write(
                "**RMSE en test:** —"
            )

        st.write(
            f"**Entrenado:** "
            f"{meta.get('fecha_entrenamiento', '—')}"
        )


# --------------------------------------------------------------------------
# PARÁMETROS
# --------------------------------------------------------------------------

st.sidebar.header("3. Parámetros del pronóstico")

pasos = st.sidebar.number_input(
    "Pasos hacia adelante a pronosticar",
    min_value=1,
    max_value=500,
    value=24,
    step=1
)


meta0 = paquetes[0][1].get("metadata", {})

frecuencia = meta0.get("frecuencia") or "h"


# --------------------------------------------------------------------------
# HISTÓRICO
# --------------------------------------------------------------------------

historicos = []

for nombre_archivo, paquete in paquetes:

    historico = extraer_historico(paquete)

    historicos.append(historico)


historico0 = historicos[0]


if historico0 is not None:

    inicio_ts = historico0.index[-1]

    st.sidebar.caption(
        f"Histórico detectado hasta **{inicio_ts}**."
    )

    n_mostrar = st.sidebar.slider(
        "Puntos de histórico a mostrar",
        min_value=min(24, len(historico0)),
        max_value=len(historico0),
        value=min(200, len(historico0))
    )

else:

    st.sidebar.warning(
        "Este .pkl no trae histórico guardado."
    )

    usar_fecha = st.sidebar.checkbox(
        "Usar fecha/hora real para el eje del gráfico",
        value=False
    )

    inicio_ts = None

    if usar_fecha:

        col_f, col_h = st.sidebar.columns(2)

        fecha_ultimo_dato = col_f.date_input(
            "Fecha del último dato real"
        )

        hora_ultimo_dato = col_h.time_input(
            "Hora del último dato real"
        )

        inicio_ts = pd.Timestamp.combine(
            fecha_ultimo_dato,
            hora_ultimo_dato
        )

    n_mostrar = 0


# --------------------------------------------------------------------------
# BOTÓN
# --------------------------------------------------------------------------

generar = st.sidebar.button(
    "🔮 Generar pronóstico",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------------------------------
# PRONÓSTICOS
# --------------------------------------------------------------------------

if generar:

    resultados = []

    for nombre_archivo, paquete in paquetes:

        etiqueta = etiqueta_modelo(
            paquete,
            nombre_archivo
        )

        try:

            predicciones = pronosticar(
                paquete,
                int(pasos)
            )

            resultados.append(
                {
                    "archivo": nombre_archivo,
                    "paquete": paquete,
                    "etiqueta": etiqueta,
                    "predicciones": predicciones
                }
            )

        except Exception as e:

            st.error(
                f"Error al pronosticar con **{etiqueta}**: {e}"
            )


    # ----------------------------------------------------------------------
    # GRÁFICAS INDIVIDUALES
    # ----------------------------------------------------------------------

    if resultados:

        st.subheader("📈 Pronósticos")


        for i, resultado in enumerate(resultados):

            paquete = resultado["paquete"]
            etiqueta = resultado["etiqueta"]
            predicciones = resultado["predicciones"]

            meta = paquete.get("metadata", {})

            variable = meta.get(
                "variable",
                ""
            )

            estacion = meta.get(
                "codigo_estacion",
                ""
            )

            historico = extraer_historico(
                paquete
            )


            # --------------------------------------------------------------
            # FECHAS DEL PRONÓSTICO
            # --------------------------------------------------------------

            if historico is not None:

                inicio_grafica = historico.index[-1]

                index_futuro = pd.date_range(
                    start=inicio_grafica,
                    periods=int(pasos) + 1,
                    freq=frecuencia
                )[1:]

            elif inicio_ts is not None:

                index_futuro = pd.date_range(
                    start=inicio_ts,
                    periods=int(pasos) + 1,
                    freq=frecuencia
                )[1:]

            else:

                index_futuro = pd.RangeIndex(
                    1,
                    int(pasos) + 1
                )


            df_resultado = pd.DataFrame(
                {
                    "Pronóstico": predicciones
                },
                index=index_futuro
            )


            # --------------------------------------------------------------
            # TÍTULO
            # --------------------------------------------------------------

            st.markdown(
                f"### 📊 Gráfica {i + 1}: {etiqueta}"
            )

            st.caption(
                f"Archivo: {resultado['archivo']}"
            )


            # --------------------------------------------------------------
            # GRÁFICA
            # --------------------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(12, 5)
            )


            if historico is not None:

                tramo_historico = historico.iloc[
                    -n_mostrar:
                ]

                ax.plot(
                    tramo_historico.index,
                    tramo_historico.values,
                    label="Histórico"
                )

                ax.axvline(
                    historico.index[-1],
                    linestyle=":"
                )


            ax.plot(
                df_resultado.index,
                df_resultado["Pronóstico"],
                label=f"Pronóstico — {etiqueta}",
                linestyle="--"
            )


            titulo = f"Pronóstico — {variable}"

            if estacion:

                titulo += (
                    f" (estación {estacion})"
                )


            ax.set_title(titulo)

            ax.set_ylabel(
                variable or "Valor"
            )

            ax.set_xlabel(
                "Fecha"
                if historico is not None or inicio_ts is not None
                else "Paso"
            )

            ax.legend()

            ax.grid(
                alpha=0.3
            )

            fig.tight_layout()

            st.pyplot(fig)


            # --------------------------------------------------------------
            # TABLA
            # --------------------------------------------------------------

            st.write(
                "🔢 **Valores pronosticados**"
            )

            st.dataframe(
                df_resultado.style.format(
                    "{:.2f}"
                ),
                use_container_width=True
            )


            # --------------------------------------------------------------
            # DESCARGA CSV
            # --------------------------------------------------------------

            csv = df_resultado.to_csv().encode(
                "utf-8"
            )

            nombre_csv = (
                f"pronostico_"
                f"{variable or 'modelo'}_"
                f"{i + 1}.csv"
            )

            st.download_button(
                "⬇️ Descargar pronóstico (CSV)",
                data=csv,
                file_name=nombre_csv,
                mime="text/csv",
                key=f"download_{i}"
            )


            if i < len(resultados) - 1:

                st.divider()


else:

    st.info(
        "Configura los parámetros en la barra lateral "
        "y presiona **Generar pronóstico**."
    )
```
