$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$testRun = 'data/test-' + [guid]::NewGuid().ToString('N')
& .venv/Scripts/python -m pytest -c backend/pyproject.toml "--basetemp=$testRun"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& .venv/Scripts/python -m ruff check backend tests
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Push-Location frontend
try {
    & npm run lint
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & npx tsc --noEmit
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & npm run build
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { Pop-Location }
