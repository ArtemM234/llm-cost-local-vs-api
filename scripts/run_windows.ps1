# Прогон на домашнем ПК (Windows + NVIDIA). Запуск из папки проекта:
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1          # пилот, 20 задач
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 -Limit 200
# Перед запуском закройте игры, браузер с видео и всё, что нагружает видеокарту.
param([int]$Limit = 20)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$cli = ".\.venv\Scripts\llmcost.exe"

if (-not (Test-Path .env) -or (Select-String -Path .env -Pattern "sk-ant-\.\.\." -Quiet)) {
    Write-Host "Впишите ANTHROPIC_API_KEY в .env (notepad .env)"; exit 1
}
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/version -TimeoutSec 5 | Out-Null }
catch { Write-Host "Ollama не запущен: откройте Ollama из меню Пуск"; exit 1 }

& $cli run --tasks gsm8k --limit $Limit --models "haiku-5.5,sonnet-5.5"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $cli run --tasks gsm8k --limit $Limit --models "qwen3-8b"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $cli analyze
