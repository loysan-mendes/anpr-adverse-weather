# Use the project verification runtime when present, otherwise an ordinary venv.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments)
$projectRoot = $PSScriptRoot
$runtimeCandidates = @(
    (Join-Path $projectRoot '.runtime311/python.exe'),
    (Join-Path $projectRoot '.venv/Scripts/python.exe')
)
$interpreter = $runtimeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $interpreter) { throw 'Create a working Python environment using README.md first.' }
$cacheRoot = Join-Path $projectRoot '.runtime_cache'
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
$env:YOLO_CONFIG_DIR = $cacheRoot
$env:PADDLE_PDX_CACHE_HOME = Join-Path $cacheRoot 'paddlex'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'
& $interpreter -B @PythonArguments
exit $LASTEXITCODE
