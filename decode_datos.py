import xarray as xr
import numpy as np
import pandas as pd

# Ruta al archivo GRIB (ajusta si lo necesitas)
file_path = "datos_clima_era5.grib"

# Abrir el archivo con cfgrib
ds = xr.open_dataset(file_path, engine="cfgrib")

# Calcular velocidad y dirección del viento
u10 = ds['u10']
v10 = ds['v10']
wind_speed = np.sqrt(u10**2 + v10**2)
wind_dir = (180/np.pi) * np.arctan2(-u10, -v10) % 360

# Añadir velocidad y dirección al dataset
ds = ds.assign(wind_speed=wind_speed)
ds = ds.assign(wind_dir=wind_dir)

# Filtrar por el área de Santa Cruz, Bolivia (aproximadamente)
# Latitudes ~ -17.0 a -18.5, Longitudes ~ -64.5 a -62.5
ds_scz = ds.sel(latitude=slice(-17.0, -18.5), longitude=slice(-64.5, -62.5))

# Convertir a DataFrame
df = ds_scz[['wind_speed', 'wind_dir']].to_dataframe().reset_index()

# Guardar como CSV
df.to_csv("viento_santa_cruz.csv", index=False)
print("Archivo guardado como viento_santa_cruz.csv")
