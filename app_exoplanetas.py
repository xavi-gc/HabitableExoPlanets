
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Exoplanetas Habitables",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Reducir padding de la página principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Reducir espacio superior */
    .main > div {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h2 class="main-header">Análisis de Exoplanetas Habitables</h2>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.nasa.gov/wp-content/themes/nasa/assets/images/nasa-logo.svg", width=200)
    st.title("Navegación")
    
    pagina = st.radio(
        "",
        ["📊 Dataset de Exoplanetas",
         "🔍 Exploración de Datos",
         "🌡️ Análisis de Temperatura",
         "⭐ Características Estelares"]
    )
    
    st.markdown("---")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    try:
        df_nasa = pd.read_csv("DataSets/exoplanetas_nasa.csv")
        return df_nasa
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo exoplanetas_nasa.csv")
        st.stop()

@st.cache_data
def procesar_datos(df_nasa):
    num_cols = [
        # Familia B - Órbita
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        # Familia C - Planeta
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_eqt",
        "pl_insol",
        # Familia D - Estrella
        "st_teff",
        "st_lum",
        "st_mass",
        "st_rad",
        "st_met",
        "st_logg",
        "st_age"
    ]
    
    id_cols = ["objectid", "pl_name", "hostname"]
    
    df_reduced = df_nasa[id_cols + num_cols].copy()
    df_reduced["objectid"] = df_reduced["objectid"].astype(str)
    
    return df_reduced, num_cols

@st.cache_data
def imputar_datos(df_reduced, num_cols):
    df_imputed = df_reduced.copy()
    
    # Imputacion: Valores por mediana
    cols_type_A = [
        "pl_rade", "pl_bmasse", "pl_dens",
        "st_teff", "st_lum", "st_mass",
        "st_rad", "st_met", "st_logg", "st_age"
    ]
    
    for col in cols_type_A:
        if col in df_imputed.columns:
            median_value = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_value)
    
    # Imputacion: Excentricidad orbital
    if "pl_orbeccen" in df_imputed.columns:
        df_imputed["pl_orbeccen"] = df_imputed["pl_orbeccen"].fillna(0)
    
    # Imputacion: Reconstrucción condicional mediante relaciones físicas explícitas
    DAYS_PER_YEAR = 365.25

    mask_orbsmax = (
        df_imputed["pl_orbsmax"].isna() &
        df_imputed["pl_orbper"].notna() &
        df_imputed["st_mass"].notna()
    )

    P_years = df_imputed.loc[mask_orbsmax, "pl_orbper"] / DAYS_PER_YEAR

    df_imputed.loc[mask_orbsmax, "pl_orbsmax"] = (
        (P_years ** (2/3)) *
        (df_imputed.loc[mask_orbsmax, "st_mass"] ** (1/3))
    )

    mask_orbper = (
        df_imputed["pl_orbper"].isna() &
        df_imputed["pl_orbsmax"].notna() &
        df_imputed["st_mass"].notna()
    )

    P_years = (
        (df_imputed.loc[mask_orbper, "pl_orbsmax"] ** (3/2)) /
        (df_imputed.loc[mask_orbper, "st_mass"] ** (1/2))
    )

    df_imputed.loc[mask_orbper, "pl_orbper"] = P_years * DAYS_PER_YEAR

    # Imputacion: Temperatura de equilibrio

    # Constante de conversión de unidades (R_sun a AU) 
    RSUN_TO_AU = 0.00465047  # 1 R☉ ≈ 0.00465047 AU (para poner R_star y a en la misma unidad)

    mask_eqt = (
        df_imputed["pl_eqt"].isna() &
        df_imputed["st_teff"].notna() &
        df_imputed["st_rad"].notna() &
        df_imputed["pl_orbsmax"].notna()
    )

    R_star_AU = df_imputed.loc[mask_eqt, "st_rad"] * RSUN_TO_AU

    df_imputed.loc[mask_eqt, "pl_eqt"] = (
        df_imputed.loc[mask_eqt, "st_teff"] *
        np.sqrt(R_star_AU / (2 * df_imputed.loc[mask_eqt, "pl_orbsmax"]))
    )

    # Imputacion: Insolación

    mask_insol = (
        df_imputed["pl_insol"].isna() &
        df_imputed["st_lum"].notna() &
        df_imputed["pl_orbsmax"].notna()
    )

    # Calculamos la insolación usando S ≈ L / a²:
    df_imputed.loc[mask_insol, "pl_insol"] = (
        (10 ** df_imputed.loc[mask_insol, "st_lum"]) /  
        (df_imputed.loc[mask_insol, "pl_orbsmax"] ** 2)
    )

    cols_problem = ["pl_orbper", "pl_orbsmax", "pl_eqt", "pl_insol"]
    df_work = df_imputed.dropna(subset=cols_problem)

    return df_work


def winsorizar_tukey(x, threshold=1.5):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    
    iqr = q3 - q1

    floor = q1 - threshold * iqr
    ceiling = q3 + threshold * iqr

    return x.clip(lower=floor, upper=ceiling)

@st.cache_data
def gestionar_outliers(df_work):

    df_wins = df_work.copy()

    for col in num_cols:
        df_wins[col] = winsorizar_tukey(df_wins[col])

    return df_wins

# Cargar datos
df_nasa = cargar_datos()
df_reduced, num_cols = procesar_datos(df_nasa)
df_work = imputar_datos(df_reduced, num_cols)
df_final = gestionar_outliers(df_work)

nombres_columnas = {
    'objectid': 'ID NASA',
    'pl_name': 'Nombre del Planeta',
    'hostname': 'Estrella',
    'pl_orbper': 'Periodo Orbital (días)',
    'pl_orbsmax': 'Distancia Orbital (AU)',
    'pl_orbeccen': 'Excentricidad Orbital',
    'pl_rade': 'Radio del Planeta (R⊕)',
    'pl_bmasse': 'Masa del Planeta (M⊕)',
    'pl_dens': 'Densidad del Planeta (g/cm³)',
    'pl_eqt': 'Temperatura de Equilibrio (K)',
    'pl_insol': 'Radiación Recibida',
    'st_teff': 'Temperatura Estelar (K)',
    'st_lum': 'Luminosidad Estelar (log)',
    'st_mass': 'Masa Estelar (M☉)',
    'st_rad': 'Radio Estelar (R☉)',
    'st_met': 'Metalicidad Estelar [Fe/H]',
    'st_logg': 'Gravedad Superficial Estelar (log g)',
    'st_age': 'Edad Estelar (Gyr)'
}

nombres_tecnicos = {v: k for k, v in nombres_columnas.items()}

# ==================== PÁGINAS ====================

if pagina == "📊 Dataset de Exoplanetas":
    # Inicializar session state para filtros dinámicos
    if 'filtros' not in st.session_state:
        st.session_state.filtros = []
    
    # ========== APLICAR FILTROS DINÁMICOS ==========
    df_filtrado = df_final.copy()
    
    # Actualizar valores de filtros desde session_state antes de aplicarlos
    for idx, filtro in enumerate(st.session_state.filtros):
        # Actualizar desde las keys de los widgets si existen
        if f"campo_{idx}" in st.session_state:
            campo_seleccionado = st.session_state[f"campo_{idx}"]
            filtro['campo'] = nombres_tecnicos.get(campo_seleccionado, campo_seleccionado)
        
        if f"operador_{idx}" in st.session_state and filtro['campo'] not in ['pl_name', 'hostname']:
            operadores = {
                'Igual a (=)': '==',
                'Mayor que (>)': '>',
                'Mayor o igual (≥)': '>=',
                'Menor que (<)': '<',
                'Menor o igual (≤)': '<='
            }
            filtro['operador'] = operadores.get(st.session_state[f"operador_{idx}"], filtro['operador'])
        
        if f"valor_{idx}" in st.session_state:
            filtro['valor'] = st.session_state[f"valor_{idx}"]
    
    # Aplicar filtros dinámicos
    for filtro in st.session_state.filtros:
        campo = filtro['campo']
        operador = filtro['operador']
        valor = filtro['valor']
        
        if campo in df_filtrado.columns:
            if operador == 'contiene':
                # Filtro de texto
                if valor:  # Solo aplicar si hay texto
                    df_filtrado = df_filtrado[
                        df_filtrado[campo].str.contains(str(valor), case=False, na=False)
                    ]
            elif operador == '==':
                df_filtrado = df_filtrado[df_filtrado[campo] == valor]
            elif operador == '>':
                df_filtrado = df_filtrado[df_filtrado[campo] > valor]
            elif operador == '>=':
                df_filtrado = df_filtrado[df_filtrado[campo] >= valor]
            elif operador == '<':
                df_filtrado = df_filtrado[df_filtrado[campo] < valor]
            elif operador == '<=':
                df_filtrado = df_filtrado[df_filtrado[campo] <= valor]
    
    # ========== MOSTRAR DATASET ==========
    if len(df_filtrado) > 0:
        # ========== TABS CON DIFERENTES VISTAS ==========
        tab1, tab2 = st.tabs(["📋 Datos", "📊 Estadísticas"])
        
        with tab1:
            # ========== SISTEMA DE FILTROS DINÁMICOS ==========
            # Título y botones de control en la misma fila
            col_titulo, col_agregar, col_limpiar = st.columns([4, 1.5, 1.5])
            with col_titulo:
                st.subheader("🔍 Filtros Avanzados")
            
            with col_agregar:
                if st.button("➕ Agregar Filtro", use_container_width=True):
                    # Crear filtro de texto por defecto (Nombre del Planeta)
                    st.session_state.filtros.append({
                        'campo': 'pl_name',
                        'operador': 'contiene',
                        'valor': ''
                    })
                    st.rerun()
            
            with col_limpiar:
                if st.button("🗑️ Limpiar Todo", use_container_width=True):
                    st.session_state.filtros = []
                    st.rerun()
            
            # Mostrar filtros existentes
            if len(st.session_state.filtros) > 0:
                st.markdown("**Filtros activos:**")
                
                filtros_a_eliminar = []
                
                for idx, filtro in enumerate(st.session_state.filtros):
                    # Determinar tipo de filtro para el resumen
                    campo_actual = filtro['campo']
                    es_texto = campo_actual in ['pl_name', 'hostname']
                    if es_texto:
                        resumen = f"🔹 Filtro {idx + 1}: {nombres_columnas.get(campo_actual, campo_actual)} contiene '{filtro['valor']}'"
                    else:
                        resumen = f"🔹 Filtro {idx + 1}: {nombres_columnas.get(campo_actual, campo_actual)} {filtro['operador']} {filtro['valor']}"
                    
                    with st.expander(resumen, expanded=True):
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            # Selector de campo (variables numéricas + columnas de texto)
                            campos_texto = ['pl_name', 'hostname']
                            campos_disponibles = campos_texto + num_cols
                            campos_disponibles_natural = [nombres_columnas.get(col, col) for col in campos_disponibles]
                            campo_actual_natural = nombres_columnas.get(filtro['campo'], filtro['campo'])
                            
                            campo_seleccionado = st.selectbox(
                                "Campo:",
                                options=campos_disponibles_natural,
                                index=campos_disponibles_natural.index(campo_actual_natural) if campo_actual_natural in campos_disponibles_natural else 0,
                                key=f"campo_{idx}"
                            )
                            nuevo_campo = nombres_tecnicos.get(campo_seleccionado, campo_seleccionado)
                            
                            # Si cambió el tipo de campo, resetear valor y operador
                            campo_anterior = filtro.get('campo')
                            es_texto_anterior = campo_anterior in ['pl_name', 'hostname']
                            es_texto_nuevo = nuevo_campo in ['pl_name', 'hostname']
                            
                            if campo_anterior != nuevo_campo and es_texto_anterior != es_texto_nuevo:
                                # Cambió el tipo de campo
                                if es_texto_nuevo:
                                    filtro['valor'] = ''
                                    filtro['operador'] = 'contiene'
                                else:
                                    filtro['valor'] = 0.0
                                    filtro['operador'] = '>='
                            
                            filtro['campo'] = nuevo_campo
                        
                        # Detectar si el campo es de texto o numérico
                        campo_tecnico = filtro['campo']
                        es_campo_texto = campo_tecnico in ['pl_name', 'hostname']
                        
                        with col2:
                            if es_campo_texto:
                                # Para texto: mostrar operador fijo "contiene"
                                st.text_input(
                                    "Operador:",
                                    value="contiene",
                                    disabled=True,
                                    key=f"operador_{idx}"
                                )
                                filtro['operador'] = 'contiene'
                            else:
                                # Para números: selector de operador
                                operadores = {
                                    'Igual a (=)': '==',
                                    'Mayor que (>)': '>',
                                    'Mayor o igual (≥)': '>=',
                                    'Menor que (<)': '<',
                                    'Menor o igual (≤)': '<='
                                }
                                
                                # Manejar caso donde el operador puede ser 'contiene' de un filtro anterior
                                operador_filtro = filtro.get('operador', '>=')
                                if operador_filtro == 'contiene':
                                    operador_filtro = '>='
                                
                                operador_actual = [k for k, v in operadores.items() if v == operador_filtro]
                                operador_actual = operador_actual[0] if operador_actual else 'Mayor o igual (≥)'
                                
                                operador_seleccionado = st.selectbox(
                                    "Operador:",
                                    options=list(operadores.keys()),
                                    index=list(operadores.keys()).index(operador_actual) if operador_actual in list(operadores.keys()) else 0,
                                    key=f"operador_{idx}"
                                )
                                filtro['operador'] = operadores[operador_seleccionado]
                        
                        with col3:
                            # Input de valor según el tipo de campo
                            if campo_tecnico in df_final.columns:
                                if es_campo_texto:
                                    # Input de texto
                                    valor = st.text_input(
                                        "Valor:",
                                        value=str(filtro.get('valor', '')),
                                        placeholder="Escribe para buscar...",
                                        key=f"valor_{idx}"
                                    )
                                    filtro['valor'] = valor
                                else:
                                    # Input numérico
                                    max_val = float(df_final[campo_tecnico].max())
                                    
                                    # Permitir valores desde 0 y usar 0 como valor por defecto
                                    # Manejar conversión segura de valor (puede ser string vacío si cambió de texto a número)
                                    try:
                                        valor_actual = float(filtro.get('valor', 0.0))
                                    except (ValueError, TypeError):
                                        valor_actual = 0.0
                                    valor_actual = max(0.0, min(max_val, valor_actual))
                                    
                                    valor = st.number_input(
                                        "Valor:",
                                        min_value=0.0,
                                        max_value=max_val,
                                        value=valor_actual,
                                        key=f"valor_{idx}",
                                        format="%.2f"
                                    )
                                    filtro['valor'] = valor
                        
                        with col4:
                            # Botón eliminar
                            if st.button("❌", key=f"eliminar_{idx}", help="Eliminar este filtro"):
                                filtros_a_eliminar.append(idx)
                
                # Eliminar filtros marcados
                if filtros_a_eliminar:
                    for idx in sorted(filtros_a_eliminar, reverse=True):
                        st.session_state.filtros.pop(idx)
                    st.rerun()
            else:
                st.info("💡 Haz clic en '➕ Agregar Filtro' para crear filtros personalizados")
            
            st.markdown("---")
            
            # Mostrar el dataframe interactivo con nombres en lenguaje natural
            df_mostrar = df_filtrado.copy()
            df_mostrar = df_mostrar.rename(columns=nombres_columnas)
            
            # Configuración para fijar solo la columna del nombre del planeta
            column_config = {
                nombres_columnas['pl_name']: st.column_config.TextColumn(
                    nombres_columnas['pl_name'],
                    width="medium",
                    pinned=True
                )
            }
            
            st.dataframe(
                df_mostrar,
                use_container_width=True,
                height=650,
                column_config=column_config,
                hide_index=True
            )

            # Header con info básica
            st.caption(f"📊 Mostrando **{len(df_filtrado)}** de {len(df_final)} exoplanetas | "
                      f"⭐ {df_filtrado['hostname'].nunique()} estrellas únicas")
        
        with tab2:
            # Usar todas las columnas numéricas disponibles
            vars_stats = num_cols
            
            if vars_stats:
                # Renombrar columnas a nombres naturales para las estadísticas
                df_stats = df_filtrado[vars_stats].describe().T
                df_stats.index = df_stats.index.map(lambda x: nombres_columnas.get(x, x))
                
                st.dataframe(
                    df_stats,
                    use_container_width=True,
                    hide_index=False
                )
                
                # Gráficos compactos de distribución
                st.markdown("---")
                num_vars = min(len(vars_stats), 12)  # Máximo 12 gráficos
                
                if num_vars > 0:
                    # Paleta de colores Okabe-Ito (amigable para daltonismo)
                    okabe_ito_colors = [
                        '#E69F00',  # Naranja
                        '#56B4E9',  # Azul cielo
                        '#009E73',  # Verde
                        '#F0E442',  # Amarillo
                        '#0072B2',  # Azul
                        '#D55E00',  # Naranja rojizo
                        '#CC79A7',  # Rosa
                    ]
                    
                    cols_per_row = 3
                    num_rows = (num_vars + cols_per_row - 1) // cols_per_row
                    
                    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 4 * num_rows))
                    axes = axes.flatten() if num_vars > 1 else [axes]
                    
                    for idx, var in enumerate(vars_stats[:12]):
                        ax = axes[idx]
                        # Usar color de la paleta Okabe-Ito (cíclico si hay más de 7 variables)
                        color = okabe_ito_colors[idx % len(okabe_ito_colors)]
                        df_filtrado[var].hist(bins=30, ax=ax, color=color, edgecolor='black', alpha=0.7)
                        # Usar nombre en lenguaje natural para el título
                        ax.set_title(nombres_columnas.get(var, var), fontweight='bold', fontsize=10)
                        ax.set_xlabel('')
                        ax.set_ylabel('Frecuencia', fontsize=9)
                    
                    for idx in range(num_vars, len(axes)):
                        axes[idx].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("💡 Selecciona variables numéricas en 'Columnas a mostrar' para ver estadísticas")
    
    else:
        st.warning("⚠️ No se encontraron resultados. Ajusta los filtros.")

elif pagina == "🔍 Exploración de Datos":
    st.subheader("Selecciona variables para analizar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_x = st.selectbox("Variable X:", num_cols, index=0)
    with col2:
        var_y = st.selectbox("Variable Y:", num_cols, index=1)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Scatter Plot", "📊 Distribuciones", "🔢 Estadísticas", "🔥 Matriz de Correlación"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_final[var_x], df_final[var_y], 
                           alpha=0.6, c=df_final['pl_eqt'], cmap='coolwarm', s=50)
        ax.set_xlabel(var_x, fontsize=12)
        ax.set_ylabel(var_y, fontsize=12)
        ax.set_title(f'{var_x} vs {var_y}', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Temperatura Equilibrio', ax=ax)
        st.pyplot(fig)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df_final[var_x].hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Distribución de {var_x}', fontweight='bold')
            ax.set_xlabel(var_x)
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            df_final[var_y].hist(bins=50, ax=ax, color='coral', edgecolor='black')
            ax.set_title(f'Distribución de {var_y}', fontweight='bold')
            ax.set_xlabel(var_y)
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Estadísticas de {var_x}:**")
            st.write(df_final[var_x].describe())
        
        with col2:
            st.write(f"**Estadísticas de {var_y}:**")
            st.write(df_final[var_y].describe())
        
        st.write(f"**Correlación:** {df_final[[var_x, var_y]].corr().iloc[0, 1]:.3f}")
    
    with tab4:
        st.subheader("🔥 Mapa de Correlación de Variables")
        
        variables_sel = st.multiselect(
            "Selecciona variables para correlación:",
            num_cols,
            default=num_cols[:8],
            key="corr_vars"
        )
        
        if len(variables_sel) >= 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = df_final[variables_sel].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, ax=ax)
            ax.set_title('Matriz de Correlación', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Selecciona al menos 2 variables")

elif pagina == "🌡️ Análisis de Temperatura":
    temp_range = st.slider(
        "Selecciona rango de temperatura (K):",
        int(df_final['pl_eqt'].min()),
        int(df_final['pl_eqt'].max()),
        (200, 400)
    )
    
    df_temp = df_final[(df_final['pl_eqt'] >= temp_range[0]) & 
                         (df_final['pl_eqt'] <= temp_range[1])]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Exoplanetas en rango", len(df_temp))
    with col2:
        st.metric("Temperatura Media", f"{df_temp['pl_eqt'].mean():.1f} K")
    with col3:
        st.metric("Desv. Estándar", f"{df_temp['pl_eqt'].std():.1f} K")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución de Temperatura")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_final['pl_eqt'], bins=50, color='orangered', edgecolor='black', alpha=0.7)
        ax.axvline(273, color='blue', linestyle='--', linewidth=2, label='Punto de congelación H₂O (273K)')
        ax.axvline(373, color='red', linestyle='--', linewidth=2, label='Punto de ebullición H₂O (373K)')
        ax.axvspan(temp_range[0], temp_range[1], alpha=0.2, color='green', label='Rango seleccionado')
        ax.set_xlabel('Temperatura de Equilibrio (K)', fontsize=12)
        ax.set_ylabel('Frecuencia', fontsize=12)
        ax.set_title('Distribución de Temperatura de Equilibrio', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🌍 Temperatura vs Radio")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_final['pl_rade'], df_final['pl_eqt'], 
                           alpha=0.6, c=df_final['pl_bmasse'], cmap='plasma', s=50)
        ax.set_xlabel('Radio del Planeta (Radio Tierra)', fontsize=12)
        ax.set_ylabel('Temperatura de Equilibrio (K)', fontsize=12)
        ax.set_title('Temperatura vs Radio del Planeta', fontsize=14, fontweight='bold')
        ax.axhline(273, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(373, color='red', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, label='Masa (Masas Tierra)', ax=ax)
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📋 Planetas en el rango seleccionado")
    st.dataframe(
        df_temp[['pl_name', 'hostname', 'pl_eqt', 'pl_rade', 'pl_bmasse', 'pl_orbsmax']].sort_values('pl_eqt'),
        use_container_width=True
    )

elif pagina == "⭐ Características Estelares":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Masa vs Radio Estelar")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_final['st_mass'], df_final['st_rad'], 
                           alpha=0.6, c=df_final['st_teff'], cmap='hot', s=50)
        ax.set_xlabel('Masa Estelar (Masas Solares)', fontsize=12)
        ax.set_ylabel('Radio Estelar (Radios Solares)', fontsize=12)
        ax.set_title('Relación Masa-Radio Estelar', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Temperatura Estelar (K)', ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🔥 Temperatura vs Luminosidad")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df_final['st_teff'], df_final['st_lum'], 
                           alpha=0.6, c=df_final['st_age'], cmap='viridis', s=50)
        ax.set_xlabel('Temperatura Estelar (K)', fontsize=12)
        ax.set_ylabel('Luminosidad (Log)', fontsize=12)
        ax.set_title('Diagrama HR (Temperatura-Luminosidad)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        plt.colorbar(scatter, label='Edad Estelar (Gyr)', ax=ax)
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Distribución Masa Estelar")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_final['st_mass'], bins=40, color='gold', edgecolor='black', alpha=0.7)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Masa Solar')
        ax.set_xlabel('Masa Estelar (M☉)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Distribución Temperatura")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_final['st_teff'], bins=40, color='orangered', edgecolor='black', alpha=0.7)
        ax.axvline(5778, color='yellow', linestyle='--', linewidth=2, label='Temp. Solar (5778K)')
        ax.set_xlabel('Temperatura (K)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)
    
    with col3:
        st.subheader("📊 Distribución Edad")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_final['st_age'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(4.6, color='orange', linestyle='--', linewidth=2, label='Edad Solar (4.6 Gyr)')
        ax.set_xlabel('Edad (Gyr)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🪐 Datos: NASA Exoplanet Archive</p>
    </div>
""", unsafe_allow_html=True)
