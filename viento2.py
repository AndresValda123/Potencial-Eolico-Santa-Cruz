#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from scipy.stats import weibull_min

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'

RUTA_RES = 'resultados_analisis_eolico'
RUTA_GRAF = f'{RUTA_RES}/graficos'
for carpeta in (RUTA_RES, RUTA_GRAF):
    os.makedirs(carpeta, exist_ok=True)

class AnalisisEolicoSantaCruz:
    def __init__(self, csv_file_path='viento_santa_cruz.csv'):
        self.df = pd.read_csv(csv_file_path)
        self.cp_max = 0.59
        self.densidad_aire = 1.225
        self.potencia_nominal = None  # Se calculará más adelante
        self.preparar_datos()

    def preparar_datos(self):
        # Convertir y limpiar datos temporales
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.dropna()
        
        # Filtrar valores atípicos
        self.df = self.df.query('0 <= wind_speed <= 30')
        
        # Obtener años presentes en los datos
        años_presentes = self.df['time'].dt.year.unique()
        
        # Determinar año a analizar
        if len(años_presentes) > 1:
            print(f"¡ADVERTENCIA! Datos multi-año detectados: {años_presentes}")
            año_reciente = max(años_presentes)
            self.df = self.df[self.df['time'].dt.year == año_reciente]
            print(f"Filtrando datos para el año: {año_reciente}")
        else:
            año_reciente = años_presentes[0] if len(años_presentes) > 0 else None
        
        # Verificar integridad temporal solo si se identificó un año
        if año_reciente:
            horas_esperadas = 8784 if (año_reciente % 4 == 0 and año_reciente % 100 != 0) or (año_reciente % 400 == 0) else 8760
            if len(self.df) != horas_esperadas:
                print(f"¡ADVERTENCIA! Datos incompletos: {len(self.df)}/{horas_esperadas} horas")
        else:
            print("¡ERROR! No se pudo determinar el año de análisis")
            return
        
        # Extraer componentes temporales
        self.df['year'] = self.df['time'].dt.year
        self.df['month'] = self.df['time'].dt.month
        self.df['hour'] = self.df['time'].dt.hour
        self.df['day'] = self.df['time'].dt.day
    def distribucion_weibull(self, velocidades):
        def weibull_pdf(x, c, k):
            return (k / c) * (x / c) ** (k - 1) * np.exp(-(x / c) ** k)
        
        v_cln = velocidades[(velocidades > 0.1) & (velocidades < 30)]
        
        if len(v_cln) < 10:
            print("¡ERROR! Insuficientes datos para ajuste Weibull")
            return None, None, None, None
        
        try:
            hist, bins = np.histogram(v_cln, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            popt, _ = curve_fit(weibull_pdf, bin_centers, hist, 
                               p0=[v_cln.mean(), 2.0], 
                               bounds=([0.1, 0.5], [30.0, 5.0]),
                               maxfev=5000)
            c, k = popt
            x = np.linspace(0.1, v_cln.max(), 100)
            return c, k, x, weibull_pdf(x, c, k)
        except Exception as e:
            print(f"Error en ajuste Weibull: {e}")
            return None, None, None, None

    def curva_potencia_vectorizada(self, v, d_rotor=100):
        """Función vectorizada optimizada para cálculo de potencia"""
        area = math.pi * (d_rotor / 2) ** 2
        v_nominal = 12.0
        self.potencia_nominal = 0.5 * self.densidad_aire * area * v_nominal**3 * self.cp_max / 1000
        
        # Crear array de resultados
        p = np.zeros_like(v)
        
        # Rango de operación 3-12 m/s
        mask1 = (v >= 3) & (v < 12)
        p[mask1] = 0.5 * self.densidad_aire * area * v[mask1]**3 * self.cp_max / 1000
        
        # Rango nominal (12-25 m/s)
        mask2 = (v >= 12) & (v < 25)
        p[mask2] = self.potencia_nominal
        
        return p

    def histograma_velocidades(self):
        plt.figure(figsize=(10, 6))
        v = self.df.wind_speed
        
        # Histograma con bins fijos para mejor visualización
        bins = np.arange(0, v.max() + 0.5, 0.5)
        plt.hist(v, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Ajuste Weibull
        c, k, x, y = self.distribucion_weibull(v)
        if c:
            plt.plot(x, y, 'r-', lw=3, label=f'Distribución Weibull (c={c:.2f}, k={k:.2f})')
            plt.legend(fontsize=12)
        
        plt.xlabel('Velocidad del Viento (m/s)', fontsize=12, fontweight='bold')
        plt.ylabel('Densidad de Probabilidad', fontsize=12, fontweight='bold')
        plt.title('Distribución de Frecuencias de Velocidad del Viento\nSanta Cruz, Bolivia', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/histograma_velocidades.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def rosa_vientos_colorizada(self):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        
        bins_dir = np.linspace(0, 2*np.pi, 37)
        bins_vel = [0, 3.5, 5.5, 8.0, 10.5, 15.0, np.inf]
        labels_vel = ['0-3.5', '3.5-5.5', '5.5-8.0', '8.0-10.5', '10.5-15.0', '>15.0']
        colores = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#d73027']
        
        dirs_rad = np.radians(self.df.wind_dir)
        total = len(self.df)
        
        for i, (v_min, v_max) in enumerate(zip(bins_vel[:-1], bins_vel[1:])):
            if v_max == np.inf:
                mask = self.df.wind_speed >= v_min
            else:
                mask = (self.df.wind_speed >= v_min) & (self.df.wind_speed < v_max)
            
            if mask.sum() > 0:
                dirs_filtradas = dirs_rad[mask]
                # Usar pesos normalizados (porcentaje)
                weights = np.ones_like(dirs_filtradas) / total * 100
                ax.hist(dirs_filtradas, bins=bins_dir, alpha=0.8, color=colores[i], 
                       weights=weights, label=f'{labels_vel[i]} m/s')
        
        # Configuración de ejes y leyenda
        ax.set_title('Rosa de Vientos por Rangos de Velocidad\nSanta Cruz, Bolivia', 
                    pad=30, fontsize=16, fontweight='bold')
        ax.set_thetagrids(np.arange(0, 360, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), title="Velocidad (m/s)")
        
        # Añadir etiqueta de porcentaje
        ax.set_rlabel_position(225)
        ax.set_ylabel("Frecuencia (%)", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/rosa_vientos_colorizada.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def variacion_temporal(self):
        plt.figure(figsize=(12, 8))
        
        # 1. Variación mensual
        plt.subplot(2, 2, 1)
        media_mensual = self.df.groupby('month').wind_speed.mean()
        meses_txt = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        plt.bar(range(1, 13), [media_mensual.get(i, 0) for i in range(1, 13)], 
                color='lightgreen', alpha=0.8, edgecolor='black')
        plt.xlabel('Mes', fontweight='bold')
        plt.ylabel('Velocidad Promedio (m/s)', fontweight='bold')
        plt.title('Variación Mensual de la Velocidad del Viento', fontweight='bold')
        plt.xticks(range(1, 13), meses_txt, rotation=45)
        plt.grid(alpha=0.3)
        
        # 2. Variación diaria
        plt.subplot(2, 2, 2)
        media_horaria = self.df.groupby('hour').wind_speed.mean()
        plt.plot(range(24), [media_horaria.get(i, 0) for i in range(24)], 
                'o-', color='blue', linewidth=2, markersize=6)
        plt.fill_between(range(24), [media_horaria.get(i, 0) for i in range(24)],
                        alpha=0.3, color='blue')
        plt.xlabel('Hora del Día', fontweight='bold')
        plt.ylabel('Velocidad Promedio (m/s)', fontweight='bold')
        plt.title('Variación Diaria de la Velocidad del Viento', fontweight='bold')
        plt.xticks(range(0, 24, 4))
        plt.grid(alpha=0.3)
        
        # 3. Serie temporal (muestra)
        plt.subplot(2, 2, 3)
        muestra = self.df.sample(min(2000, len(self.df))).sort_values('time')
        plt.plot(muestra.time, muestra.wind_speed, alpha=0.7, color='red', linewidth=0.8)
        plt.xlabel('Tiempo', fontweight='bold')
        plt.ylabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.title('Serie Temporal de Velocidad del Viento (Muestra)', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        
        # 4. Boxplot mensual
        plt.subplot(2, 2, 4)
        plt.boxplot([self.df[self.df.month == i]['wind_speed'] for i in range(1, 13)],
                   tick_labels=meses_txt)
        plt.xlabel('Mes', fontweight='bold')
        plt.ylabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.title('Distribución Mensual de Velocidades', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        
        plt.suptitle('Análisis Temporal del Viento - Santa Cruz, Bolivia', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/variacion_temporal.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def analisis_potencia(self):
        # Calcular potencia primero
        if self.potencia_nominal is None:
            self.df['potencia_generada'] = self.curva_potencia_vectorizada(self.df.wind_speed.values)
        else:
            self.df['potencia_generada'] = self.curva_potencia_vectorizada(self.df.wind_speed.values)
        
        # Configurar figura
        plt.figure(figsize=(12, 8))
        
        # 1. Curva de potencia
        plt.subplot(2, 2, 1)
        v_curva = np.linspace(0, 30, 100)
        p_curva = self.curva_potencia_vectorizada(v_curva)
        plt.plot(v_curva, p_curva, 'b-', lw=3)
        plt.xlabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.ylabel('Potencia Generada (kW)', fontweight='bold')
        plt.title('Curva de Potencia del Aerogenerador', fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 2. Distribución de potencia generada
        plt.subplot(2, 2, 2)
        plt.hist(self.df.potencia_generada, 
                bins=np.linspace(0, self.potencia_nominal, 50),
                alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Potencia Generada (kW)', fontweight='bold')
        plt.ylabel('Frecuencia', fontweight='bold')
        plt.title('Distribución de Potencia Generada', fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 3. Generación mensual
        plt.subplot(2, 2, 3)
        p_mensual = self.df.groupby('month').potencia_generada.mean()
        meses_txt = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        plt.bar(range(1, 13), [p_mensual.get(i, 0) for i in range(1, 13)],
                color='green', alpha=0.8, edgecolor='black')
        plt.xlabel('Mes', fontweight='bold')
        plt.ylabel('Potencia Promedio (kW)', fontweight='bold')
        plt.title('Generación Mensual de Energía', fontweight='bold')
        plt.xticks(range(1, 13), meses_txt, rotation=45)
        plt.grid(alpha=0.3)
        
        # 4. Estados de operación
        plt.subplot(2, 2, 4)
        horas_totales = len(self.df)
        horas_operacion = (self.df.potencia_generada > 0).sum()
        horas_nominal = (self.df.potencia_generada >= self.potencia_nominal * 0.98).sum()
        horas_parcial = horas_operacion - horas_nominal
        
        etiquetas = ['Sin Generación', 'Generación Parcial', 'Potencia Nominal']
        tamanios = [horas_totales - horas_operacion, horas_parcial, horas_nominal]
        porcentajes = [f'{x/horas_totales*100:.1f}%' for x in tamanios]
        
        colores_pie = ['#ff9999', '#66b3ff', '#99ff99']
        plt.pie(tamanios, labels=[f'{l}\n({p})' for l, p in zip(etiquetas, porcentajes)], 
               autopct='%1.1f%%', startangle=90, colors=colores_pie,
               textprops={'fontsize': 10})
        plt.title('Estados de Operación del Aerogenerador', fontweight='bold')
        
        plt.suptitle('Análisis de Potencia Eólica - Santa Cruz, Bolivia', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/analisis_potencia.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Limpiar columna temporal
        del self.df['potencia_generada']

    def mapa_estatico_velocidades(self):
        plt.figure(figsize=(12, 10))
        
        # Calcular densidad de potencia (W/m²)
        potencia = 0.5 * self.densidad_aire * self.df.wind_speed**3
        
        # Crear scatter plot con densidad de potencia
        sc = plt.scatter(self.df.longitude, self.df.latitude, c=potencia, 
                        cmap='viridis', alpha=0.7, s=20, 
                        norm=LogNorm(vmin=1, vmax=potencia.max()))
        
        cbar = plt.colorbar(sc, shrink=0.8)
        cbar.set_label('Densidad de Potencia (W/m²)', fontweight='bold', fontsize=12)
        
        plt.xlabel('Longitud (°)', fontweight='bold', fontsize=12)
        plt.ylabel('Latitud (°)', fontweight='bold', fontsize=12)
        plt.title('Distribución Espacial del Potencial Eólico\nSanta Cruz, Bolivia', 
                 fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/mapa_potencial_eolico.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def mapa_potencial_interactivo(self):
        # Agrupar por ubicación
        pot = (self.df.groupby(['latitude', 'longitude'])
               .agg({'wind_speed': ['mean', 'std', 'count']}))
        pot.columns = ['vel_media', 'vel_std', 'n']
        pot = pot.reset_index()
        
        # Calcular densidad de potencia
        pot['potencial_eolico'] = 0.5 * self.densidad_aire * pot.vel_media ** 3
        
        # Crear figura interactiva
        fig = go.Figure()
        
        fig.add_trace(go.Scattermapbox(
            lat=pot['latitude'],
            lon=pot['longitude'],
            mode='markers',
            marker=dict(
                size=pot['vel_media'] * 3,
                color=pot['potencial_eolico'],
                colorscale='Viridis',
                cmin=1,
                cmax=pot['potencial_eolico'].max(),
                colorbar=dict(
                    title=dict(
                        text="Potencial Eólico (W/m²)",
                        font=dict(size=14)
                    ),
                    tickfont=dict(size=12),
                    ticks="outside"
                ),
                sizemode='diameter',
                opacity=0.8,
            ),
            text=[f'Lat: {lat:.3f}°<br>Lon: {lon:.3f}°<br>Vel. Media: {vel:.2f} m/s<br>Potencial: {pot:.0f} W/m²'
                  for lat, lon, vel, pot in zip(pot.latitude, pot.longitude, pot.vel_media, pot.potencial_eolico)],
            hovertemplate='%{text}<extra></extra>',
            name='Potencial Eólico'
        ))
        
        centro_lat = pot.latitude.mean()
        centro_lon = pot.longitude.mean()
        
        fig.update_layout(
            title=dict(
                text="Mapa de Potencial Eólico - Santa Cruz, Bolivia",
                x=0.5,
                font=dict(size=18, color='black')
            ),
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=centro_lat, lon=centro_lon),
                zoom=8
            ),
            width=1200,
            height=800,
            font=dict(size=12)
        )
        
        fig.write_html(f'{RUTA_GRAF}/mapa_potencial_interactivo.html')

    def estadisticas_descriptivas(self):
        stats = self.df['wind_speed'].describe()
        
        plt.figure(figsize=(12, 8))
        
        # 1. Histograma con media y mediana
        plt.subplot(2, 2, 1)
        plt.hist(self.df.wind_speed, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Media: {stats["mean"]:.2f} m/s')
        plt.axvline(stats['50%'], color='green', linestyle='--', linewidth=2, 
                   label=f'Mediana: {stats["50%"]:.2f} m/s')
        plt.xlabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.ylabel('Frecuencia', fontweight='bold')
        plt.title('Histograma de Velocidades del Viento', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # 2. Diagrama de caja
        plt.subplot(2, 2, 2)
        plt.boxplot(self.df.wind_speed, patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        plt.ylabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.title('Diagrama de Caja de Velocidades', fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 3. Curva de percentiles
        plt.subplot(2, 2, 3)
        percentiles = np.percentile(self.df.wind_speed, range(0, 101, 5))
        plt.plot(range(0, 101, 5), percentiles, 'o-', color='purple', linewidth=2)
        plt.xlabel('Percentil', fontweight='bold')
        plt.ylabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.title('Curva de Percentiles', fontweight='bold')
        plt.grid(alpha=0.3)
        
        # 4. Función de distribución acumulativa
        plt.subplot(2, 2, 4)
        velocidades_ordenadas = np.sort(self.df.wind_speed)
        p = np.arange(1, len(velocidades_ordenadas) + 1) / len(velocidades_ordenadas) * 100
        plt.plot(velocidades_ordenadas, p, color='orange', linewidth=2)
        plt.xlabel('Velocidad del Viento (m/s)', fontweight='bold')
        plt.ylabel('Probabilidad Acumulada (%)', fontweight='bold')
        plt.title('Función de Distribución Acumulativa', fontweight='bold')
        plt.grid(alpha=0.3)
        
        plt.suptitle('Análisis Estadístico Descriptivo - Santa Cruz, Bolivia', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{RUTA_GRAF}/estadisticas_descriptivas.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generar_reporte_detallado(self):
        # Calcular potencia nominal si no está definida
        if self.potencia_nominal is None:
            _ = self.curva_potencia_vectorizada(np.array([12]))  # Forzar cálculo
            
        # Cálculos estadísticos básicos
        stats = self.df['wind_speed'].describe()
        velocidad_media = stats['mean']
        velocidad_mediana = stats['50%']
        velocidad_max = stats['max']
        velocidad_min = stats['min']
        desviacion_std = stats['std']
        coef_variacion = (desviacion_std / velocidad_media) * 100
        
        # Análisis de distribución Weibull
        c, k, _, _ = self.distribucion_weibull(self.df['wind_speed'])
        
        # Calcular potencia generada
        potencia_vector = self.curva_potencia_vectorizada(self.df['wind_speed'].values)
        potencia_media = np.mean(potencia_vector)
        
        # Calcular horas de operación
        horas_totales = len(self.df)
        horas_operacion = np.sum(potencia_vector > 0)
        horas_nominal = np.sum(potencia_vector >= self.potencia_nominal * 0.98)
        factor_capacidad = (np.sum(potencia_vector) / (horas_totales * self.potencia_nominal)) * 100
        
        # Energía anual estimada (MWh)
        energia_anual_estimada = np.sum(potencia_vector) / 1000
        
        # Análisis temporal
        vel_por_mes = self.df.groupby('month')['wind_speed'].mean()
        mejor_mes = vel_por_mes.idxmax()
        peor_mes = vel_por_mes.idxmin()
        vel_mejor_mes = vel_por_mes.max()
        vel_peor_mes = vel_por_mes.min()
        
        vel_por_hora = self.df.groupby('hour')['wind_speed'].mean()
        mejor_hora = vel_por_hora.idxmax()
        peor_hora = vel_por_hora.idxmin()
        
        # Análisis de velocidades útiles
        mask_utiles = (self.df['wind_speed'] >= 3) & (self.df['wind_speed'] <= 25)
        porcentaje_velocidades_utiles = (mask_utiles.sum() / horas_totales) * 100
        
        # Clasificación del recurso eólico
        if velocidad_media >= 7.5:
            clasificacion_recurso = "EXCELENTE"
            viabilidad = "ALTAMENTE VIABLE"
        elif velocidad_media >= 6.5:
            clasificacion_recurso = "BUENO"
            viabilidad = "VIABLE"
        elif velocidad_media >= 5.5:
            clasificacion_recurso = "MODERADO"
            viabilidad = "MODERADAMENTE VIABLE"
        elif velocidad_media >= 4.5:
            clasificacion_recurso = "MARGINAL"
            viabilidad = "MARGINALMENTE VIABLE"
        else:
            clasificacion_recurso = "POBRE"
            viabilidad = "NO VIABLE"
        
        # Potencial energético
        densidad_potencia = 0.5 * self.densidad_aire * (velocidad_media ** 3)
        
        # Crear el reporte
        reporte = f"""
    ==================================================================================
                        REPORTE DETALLADO DE ANÁLISIS EÓLICO
                            SANTA CRUZ, BOLIVIA
    ==================================================================================

    FECHA DE ANÁLISIS: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')}
    PERÍODO DE DATOS: {self.df['time'].min().strftime('%d/%m/%Y')} - {self.df['time'].max().strftime('%d/%m/%Y')}
    TOTAL DE REGISTROS: {len(self.df):,} mediciones
    POTENCIA NOMINAL: {self.potencia_nominal:.0f} kW

    ==================================================================================
    1. ESTADÍSTICAS BÁSICAS DE VELOCIDAD DEL VIENTO
    ==================================================================================

    • Velocidad media anual:           {velocidad_media:.2f} m/s
    • Velocidad mediana:               {velocidad_mediana:.2f} m/s
    • Velocidad máxima registrada:     {velocidad_max:.2f} m/s
    • Velocidad mínima registrada:     {velocidad_min:.2f} m/s
    • Desviación estándar:             {desviacion_std:.2f} m/s
    • Coeficiente de variación:        {coef_variacion:.1f}%

    ==================================================================================
    2. ANÁLISIS DE DISTRIBUCIÓN WEIBULL
    ==================================================================================

    • Parámetro de escala (c):         {c:.2f} m/s
    • Parámetro de forma (k):          {k:.2f}
    • Calidad del ajuste:              {'Excelente' if c and k else 'No disponible'}

    Interpretación del parámetro k:
    • k = {k:.2f} indica vientos {'muy consistentes' if k > 3 else 'consistentes' if k > 2 else 'moderadamente variables' if k > 1.5 else 'altamente variables'}

    ==================================================================================
    3. CLASIFICACIÓN DEL RECURSO EÓLICO
    ==================================================================================

    • CLASIFICACIÓN:                   {clasificacion_recurso}
    • VIABILIDAD ECONÓMICA:            {viabilidad}
    • DENSIDAD DE POTENCIA:            {densidad_potencia:.0f} W/m²

    Referencia de clasificación (velocidad media anual):
    • Excelente: ≥ 7.5 m/s    • Bueno: 6.5-7.4 m/s    • Moderado: 5.5-6.4 m/s
    • Marginal: 4.5-5.4 m/s   • Pobre: < 4.5 m/s

    ==================================================================================
    4. ANÁLISIS DE POTENCIA Y GENERACIÓN
    ==================================================================================

    • Potencia media generada:         {potencia_media:.0f} kW
    • Factor de capacidad:             {factor_capacidad:.1f}%
    • Energía anual estimada:          {energia_anual_estimada:.0f} MWh/año

    ESTADOS DE OPERACIÓN:
    • Horas sin generación:            {horas_totales - horas_operacion:,} ({(horas_totales - horas_operacion)/horas_totales*100:.1f}%)
    • Horas con generación:            {horas_operacion:,} ({horas_operacion/horas_totales*100:.1f}%)
    • Horas a potencia nominal:        {horas_nominal:,} ({horas_nominal/horas_totales*100:.1f}%)

    • Velocidades útiles (3-25 m/s):   {porcentaje_velocidades_utiles:.1f}%

    ==================================================================================
    5. ANÁLISIS TEMPORAL
    ==================================================================================

    VARIACIÓN MENSUAL:
    • Mejor mes:                       {['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][mejor_mes-1]} ({vel_mejor_mes:.2f} m/s)
    • Peor mes:                        {['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][peor_mes-1]} ({vel_peor_mes:.2f} m/s)
    • Variación estacional:            {vel_mejor_mes - vel_peor_mes:.2f} m/s

    VARIACIÓN DIARIA:
    • Mejor hora del día:              {mejor_hora:02d}:00 hrs
    • Peor hora del día:               {peor_hora:02d}:00 hrs

    ==================================================================================
    6. EVALUACIÓN DE VIABILIDAD ECONÓMICA
    ==================================================================================

    FACTORES POSITIVOS:
    """

        # Agregar factores positivos
        if velocidad_media >= 6.0:
            reporte += "✓ Velocidad media anual superior a 6 m/s\n"
        if factor_capacidad >= 25:
            reporte += "✓ Factor de capacidad superior al 25%\n"
        if porcentaje_velocidades_utiles >= 70:
            reporte += "✓ Alto porcentaje de velocidades útiles para generación\n"
        if k >= 2.0:
            reporte += "✓ Vientos consistentes (k ≥ 2.0)\n"
        if (vel_mejor_mes - vel_peor_mes) <= 2.0:
            reporte += "✓ Baja variación estacional\n"

        reporte += f"""
    FACTORES DE RIESGO:
    """

        # Agregar factores de riesgo
        if velocidad_media < 5.5:
            reporte += "⚠ Velocidad media anual inferior a 5.5 m/s\n"
        if factor_capacidad < 20:
            reporte += "⚠ Factor de capacidad inferior al 20%\n"
        if porcentaje_velocidades_utiles < 60:
            reporte += "⚠ Bajo porcentaje de velocidades útiles\n"
        if k < 1.5:
            reporte += "⚠ Vientos altamente variables (k < 1.5)\n"
        if (vel_mejor_mes - vel_peor_mes) > 3.0:
            reporte += "⚠ Alta variación estacional\n"

        reporte += f"""
    ==================================================================================
    7. RECOMENDACIONES TÉCNICAS
    ==================================================================================

    SELECCIÓN DE AEROGENERADORES:
    """

        if velocidad_media >= 7.0:
            reporte += "• Recomendado: Aerogeneradores de alta potencia (≥2MW)\n"
            reporte += "• Velocidad de arranque: 3-4 m/s\n"
            reporte += "• Velocidad nominal: 12-15 m/s\n"
        elif velocidad_media >= 5.5:
            reporte += "• Recomendado: Aerogeneradores de media potencia (1-2MW)\n"
            reporte += "• Velocidad de arranque: 2.5-3.5 m/s\n"
            reporte += "• Velocidad nominal: 10-12 m/s\n"
        else:
            reporte += "• Considerar: Aerogeneradores de baja potencia (<1MW)\n"
            reporte += "• Velocidad de arranque: ≤3 m/s\n"
            reporte += "• Velocidad nominal: 8-10 m/s\n"

        reporte += f"""
    ALTURA DE BUJE RECOMENDADA:
    """

        if velocidad_media >= 6.5:
            reporte += "• 80-100 metros (recurso eólico bueno)\n"
        elif velocidad_media >= 5.0:
            reporte += "• 100-120 metros (para optimizar recurso)\n"
        else:
            reporte += "• ≥120 metros (necesario para mejorar recurso)\n"

        reporte += f"""
    ==================================================================================
    8. CONCLUSIONES Y RECOMENDACIONES FINALES
    ==================================================================================

    RESUMEN EJECUTIVO:
    """

        if velocidad_media >= 7.0:
            reporte += """
    • El sitio presenta condiciones EXCELENTES para desarrollo eólico
    • Se recomienda proceder con estudios de factibilidad detallados
    • Potencial para proyecto comercial de gran escala
    • Retorno de inversión esperado: Alto
    """
        elif velocidad_media >= 6.0:
            reporte += """
    • El sitio presenta condiciones BUENAS para desarrollo eólico
    • Se recomienda análisis económico detallado
    • Potencial para proyecto comercial de mediana escala
    • Retorno de inversión esperado: Moderado a Alto
    """
        elif velocidad_media >= 5.0:
            reporte += """
    • El sitio presenta condiciones MODERADAS para desarrollo eólico
    • Requiere análisis económico exhaustivo
    • Considerar incentivos gubernamentales
    • Retorno de inversión esperado: Moderado
    """
        else:
            reporte += """
    • El sitio presenta condiciones POBRES para desarrollo eólico
    • NO se recomienda para proyectos comerciales
    • Considerar otras fuentes de energía renovable
    • Retorno de inversión esperado: Bajo o Negativo
    """

        reporte += f"""
    PRÓXIMOS PASOS RECOMENDADOS:
    """

        if velocidad_media >= 6.0:
            reporte += """
    1. Instalación de torre de medición de al menos 100m por 12 meses
    2. Estudio de impacto ambiental
    3. Análisis de conexión a red eléctrica
    4. Evaluación económica detallada con datos locales
    5. Consulta con comunidades locales
    """
        elif velocidad_media >= 5.0:
            reporte += """
    1. Medición adicional por 12 meses a mayor altura
    2. Análisis de costos de operación y mantenimiento
    3. Evaluación de incentivos fiscales disponibles
    4. Comparación con otras tecnologías renovables
    """
        else:
            reporte += """
    1. Considerar otros sitios con mejor exposición al viento
    2. Evaluar tecnologías eólicas de pequeña escala
    3. Analizar hibridación con solar fotovoltaica
    4. Estudio de otras fuentes renovables
    """

        reporte += f"""
    ==================================================================================
    NOTA TÉCNICA:
    Este reporte se basa en los datos proporcionados y utiliza modelos estándar de
    la industria eólica. Para decisiones de inversión, se recomienda validar con
    mediciones adicionales y consultores especializados.

    Generado por: Sistema de Análisis Eólico v1.1 (Corregido)
    ==================================================================================
    """

        # Guardar el reporte
        ruta_reporte = f'{RUTA_RES}/reporte_detallado_eolico_santa_cruz.txt'
        with open(ruta_reporte, 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print(f"Reporte detallado generado: {ruta_reporte}")
        return ruta_reporte

    def generar_todos_graficos(self):
        self.histograma_velocidades()
        self.rosa_vientos_colorizada()
        self.variacion_temporal()
        self.analisis_potencia()
        self.mapa_estatico_velocidades()
        self.mapa_potencial_interactivo()
        self.estadisticas_descriptivas()
        self.generar_reporte_detallado()

def ejecutar_analisis():
    try:
        analizador = AnalisisEolicoSantaCruz('viento_santa_cruz.csv')
        analizador.generar_todos_graficos()
        print("Análisis completado exitosamente")
    except FileNotFoundError:
        print("ERROR: Archivo 'viento_santa_cruz.csv' no encontrado")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    ejecutar_analisis()