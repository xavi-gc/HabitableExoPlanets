# Script para instalar dependencias y ejecutar la aplicación Streamlit
# Python 3.13.5

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Instalación de Exoplanetas Streamlit App" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Verificar versión de Python
Write-Host "Verificando versión de Python..." -ForegroundColor Yellow
py --version

Write-Host ""
Write-Host "Instalando dependencias..." -ForegroundColor Yellow
Write-Host ""

# Instalar dependencias
py -m pip install --upgrade pip
py -m pip install streamlit pandas numpy matplotlib seaborn

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Instalación completada!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para ejecutar la aplicación, usa:" -ForegroundColor Cyan
Write-Host "  .\run_app.ps1" -ForegroundColor White
Write-Host ""
