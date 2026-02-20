import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# Cargar el dataset de Iris desde scikit-learn y devolver un DataFrame
def load_iris_dataframe():
    iris_bunch = load_iris()
    df_iris = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)
    df_iris['target'] = iris_bunch.target
    return df_iris


def main():
    st.set_page_config(page_title="EDA - Iris", layout="wide")
    st.title("Exploratory Data Analysis — Iris Dataset")

    # Cargar datos
    df = load_iris_dataframe()

    # Mostrar primeras filas
    st.header("Primeras filas del dataset")
    st.dataframe(df.head())

    # Mostrar estadísticas descriptivas
    st.header("Estadísticas resumen")
    st.dataframe(df.describe())

    # Columnas numéricas disponibles
    numeric_columns = df.select_dtypes(include='number').columns.tolist()

    # Panel lateral para controles de usuario
    st.sidebar.header("Controles de visualización")
    selected_columns = st.sidebar.multiselect(
        "Selecciona columnas numéricas para graficar",
        options=numeric_columns,
        default=numeric_columns,
    )

    if not selected_columns:
        st.warning("Selecciona al menos una columna numérica desde la barra lateral.")
        return

    # Histograma(s)
    st.subheader("Histograma(s)")
    num_selected = len(selected_columns)
    fig_hist, axes = plt.subplots(nrows=1, ncols=num_selected, figsize=(4 * num_selected, 4))
    if num_selected == 1:
        axes = [axes]
    for ax, col in zip(axes, selected_columns):
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(f"Distribución — {col}")
    st.pyplot(fig_hist)

    # Scatter plot: elegir ejes X e Y
    st.subheader("Diagrama de dispersión")
    default_x = selected_columns[0]
    default_y = selected_columns[1] if len(selected_columns) > 1 else selected_columns[0]
    x_axis = st.sidebar.selectbox("Eje X", options=selected_columns, index=selected_columns.index(default_x))
    y_axis = st.sidebar.selectbox("Eje Y", options=selected_columns, index=selected_columns.index(default_y))

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='target', palette='deep', ax=ax_scatter)
    ax_scatter.set_title(f"Scatter: {x_axis} vs {y_axis}")
    st.pyplot(fig_scatter)


if __name__ == "__main__":
    main()
