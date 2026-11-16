# Run from any directory; the server remains attached to this console.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$baseUrl = 'http://127.0.0.1:8000'

function Test-EduRisk {
    try {
        $health = Invoke-RestMethod -Uri "$baseUrl/api/health" -TimeoutSec 2
        return ($health.status -eq 'ok' -and $health.mode -eq 'SYNTHETIC_DEMO_ONLY' -and $health.version)
    } catch { return $false }
}

if (Test-EduRisk) {
    Write-Host "EduRisk уже работает: $baseUrl" -ForegroundColor Green
    try { Start-Process $baseUrl } catch { Write-Host "Откройте этот адрес в браузере: $baseUrl" }
    exit 0
}

$portClient = New-Object System.Net.Sockets.TcpClient
$portBusy = $false
try { $portClient.Connect('127.0.0.1', 8000); $portBusy = $true } catch {} finally { $portClient.Dispose() }
if ($portBusy) {
    Write-Host 'Порт 8000 занят, но работающий EduRisk не обнаружен.' -ForegroundColor Yellow
    Write-Host 'Если EduRisk только запускается, повторите запуск через несколько секунд.'
    Write-Host 'Иначе остановите другое приложение, использующее этот порт. Скрипт не завершает чужие процессы.'
    exit 1
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    Write-Host 'Окружение Python для проекта ещё не создано.' -ForegroundColor Yellow
    Write-Host 'Установите Python 3.11 или 3.12 с https://www.python.org/downloads/windows/.'
    Write-Host 'Затем откройте PowerShell в папке проекта и выполните:'
    Write-Host '  py -3.12 -m venv .venv'
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    Write-Host 'Для Python 3.11 замените -3.12 на -3.11. После установки снова откройте start.cmd.'
    exit 1
}

& $pythonPath -c 'import uvicorn, fastapi, pandas' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Не удалось загрузить зависимости проекта.' -ForegroundColor Yellow
    Write-Host 'В папке проекта выполните: .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
    exit 1
}

$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
Write-Host "Запускаем EduRisk: $baseUrl" -ForegroundColor Green
Write-Host 'Браузер откроется после готовности сервера. Для остановки нажмите Ctrl+C.'
$launchCode = @'
import json
import threading
import time
import urllib.request
import webbrowser
import uvicorn


def open_when_ready():
    for _ in range(40):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1) as response:
                health = json.load(response)
            if health.get('status') == 'ok' and health.get('mode') == 'SYNTHETIC_DEMO_ONLY':
                webbrowser.open('http://127.0.0.1:8000')
                return
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=open_when_ready, daemon=True).start()
uvicorn.run('edurisk.api:app', host='127.0.0.1', port=8000)
'@
& $pythonPath -c $launchCode
exit $LASTEXITCODE
