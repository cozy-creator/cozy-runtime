# Installed CLI proof

`run.sh` builds the current Runtime wheel, installs it and this directory's ordinary
endpoint fixture into a fresh environment, changes to `/`, unsets `PYTHONPATH`, cuts
network connects, and invokes only the installed `cozy-runtime` executable.

It proves the process exit matrix, noninteractive/read-only AXI, exact Descriptor/1 bytes,
public key classification, typed mixed-media input grants, canonical invocation identity,
and weightless offline execution. Real SDXL CLI-versus-worker identity remains hardware
qualification under `qualification/gpu/`.
