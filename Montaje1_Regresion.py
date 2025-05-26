import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
import pandas as pd
import openpyxl

# Activar estilo de seaborn
sns.set(style="whitegrid")

# Datos experimentales
datos = {
    'R1': {
        'V': [1.880, 3.950, 5.800, 7.540, 9.860],
        'I': [0.715, 1.450, 2.160, 2.960, 3.720]
    },
    'R2': {
        'V': [2.040, 3.920, 6.070, 7.960, 9.560],
        'I': [2.480, 4.760, 7.360, 9.660, 11.970]
    },
    'R3': {
        'V': [2.030, 3.950, 5.990, 8.020, 9.980],
        'I': [3.020, 5.880, 8.900, 11.940, 14.830]
    },
    'R4': {
        'V': [1.980, 4.080, 6.100, 8.040, 9.950],
        'I': [1.000, 2.060, 3.080, 4.070, 5.030]
    }
}

# Colores más suaves y diferentes
colores = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

# Crear la figura
plt.figure(figsize=(12, 8))

for i, (resistencia, valores) in enumerate(datos.items()):
    V = np.array(valores['V'])
    I = np.array(valores['I'])  # Convertimos a mA

    # Ajuste lineal
    pendiente, intercepto, r_value, _, _ = linregress(V, I)
    V_fit = np.linspace(min(V), max(V), 200)
    I_fit = pendiente * V_fit + intercepto

    # Puntos experimentales
    plt.scatter(V, I, color=colores[i], s=60, edgecolors='black', zorder=3)

    # Línea de ajuste
    plt.plot(V_fit, I_fit, linestyle='--', linewidth=2, color=colores[i],
             label=f'{resistencia}: I = {pendiente:.4f}·V + {intercepto:.4f}  |  R² = {r_value**2:.6f}')

# Estética del gráfico
plt.title('Gráfica Voltaje vs Corriente para cada Resistencia', fontsize=16, fontweight='bold')
plt.xlabel('Voltaje (V)', fontsize=14)
plt.ylabel('Corriente (mA)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Ajuste lineal por resistencia", fontsize=10, title_fontsize=12, loc='upper left', frameon=True, edgecolor='gray')
plt.tight_layout()

# Mostrar
plt.show()


# Crear lista para almacenar datos de la tabla
resumen = []

for resistencia, valores in datos.items():
    V = np.array(valores['V'])
    I = np.array(valores['I'])

    pendiente, intercepto, r_value, _, _ = linregress(V, I)
    resistencia_calc = 1 / pendiente if pendiente != 0 else np.nan

    ecuacion = f'I = {pendiente:.4f}·V + {intercepto:.4f}'
    R2 = r_value**2

    resumen.append({
        'Resistencia': resistencia,
        'Ecuación': ecuacion,
        'R^2': f'{R2:.6f}',
        'Resistencia calculada (Ω)': f'{resistencia_calc:.4f}'
    })

# Crear DataFrame
df_resumen = pd.DataFrame(resumen)

df_resumen.to_excel('resumen_resistencias.xlsx', index=False)
print("Archivo Excel 'resumen_resistencias.xlsx' guardado.")
