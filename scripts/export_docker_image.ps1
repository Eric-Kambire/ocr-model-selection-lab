param(
    [string]$Image = "ocr-benchmark-suite:1.0.0",
    [string]$Output = "dist/ocr-benchmark-suite-1.0.0.tar"
)

$ErrorActionPreference = "Stop"
$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Output))
$workspace = [System.IO.Path]::GetFullPath((Get-Location))

if (-not $outputPath.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The output must remain inside the workspace."
}

$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

docker image inspect $Image | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image not found: $Image"
}

docker save --output $outputPath $Image
if ($LASTEXITCODE -ne 0) {
    throw "docker save failed."
}

Write-Output "Image exported to $outputPath"
Write-Output "Recipient command: docker load --input $([System.IO.Path]::GetFileName($outputPath))"
