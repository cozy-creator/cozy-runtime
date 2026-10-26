param([Parameter(Mandatory = $true)][string]$Destination)
$ErrorActionPreference = 'Stop'
if (-not $env:COZY_CUDA_TOOLKIT) {
    throw 'COZY_CUDA_TOOLKIT must name the build-time CUDA 13 toolkit'
}
$root = Split-Path $PSScriptRoot
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "$root\native\execution\target" }
$env:NVCC = "$env:COZY_CUDA_TOOLKIT\bin\nvcc.exe"
& cargo build --locked --release --manifest-path "$root\native\execution\Cargo.toml"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$package = Join-Path $Destination 'cozy_runtime\_kernels'
New-Item -ItemType Directory -Force $package | Out-Null
Copy-Item "$target\release\_execution.dll" "$package\_execution.pyd"
