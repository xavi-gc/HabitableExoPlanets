# Script para ejecutar la aplicación Streamlit
# Python 3.13.5

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  🪐 Iniciando Aplicación de Exoplanetas" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "La aplicación se abrirá en tu navegador..." -ForegroundColor Yellow
Write-Host "URL: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Presiona Ctrl+C para detener el servidor" -ForegroundColor Yellow
Write-Host ""

# Ejecutar Streamlit
py -m streamlit run app_exoplanetas.py
