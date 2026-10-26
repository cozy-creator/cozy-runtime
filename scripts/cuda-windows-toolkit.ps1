param([string]$Root = "$env:RUNNER_TEMP\cuda")
$ErrorActionPreference = 'Stop'
$redist = 'https://developer.download.nvidia.com/compute/cuda/redist'
if (-not $env:CUDA_REDIST) { throw 'CUDA_REDIST must name the pinned toolkit manifest' }
$manifest = Invoke-RestMethod "$redist/redistrib_$env:CUDA_REDIST.json"
New-Item -ItemType Directory -Force $Root | Out-Null
foreach ($name in 'cuda_nvcc', 'cuda_crt', 'libnvvm', 'cuda_cudart', 'cuda_cccl', 'cuda_cuobjdump') {
    $entry = $manifest.$name.'windows-x86_64'
    $archive = Join-Path $env:RUNNER_TEMP "$name.zip"
    $expanded = Join-Path $env:RUNNER_TEMP "cuda-redist-$name"
    Invoke-WebRequest "$redist/$($entry.relative_path)" -OutFile $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $entry.sha256.ToUpper()) {
        throw "$name digest differs from NVIDIA's pinned manifest"
    }
    Expand-Archive $archive $expanded -Force
    Copy-Item "$expanded\*\*" $Root -Recurse -Force
}
"$Root\bin" >> $env:GITHUB_PATH
"CUDAToolkit_ROOT=$Root" >> $env:GITHUB_ENV
"CUDA_PATH=$Root" >> $env:GITHUB_ENV
"COZY_CUDA_TOOLKIT=$Root" >> $env:GITHUB_ENV
