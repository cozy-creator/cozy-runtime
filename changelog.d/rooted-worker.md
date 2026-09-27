`cozy-runtime-worker` runs on any machine under the root its Host names in
`COZY_MACHINE_ROOT` (absent means `/`), as root or as an ordinary user. Started as
root it isolates executors under their own uid as before; started as a user it runs
them as that user. It measures the GPU inventory from the driver even when its own
Python has no torch, and reports the devices, backend and executor isolation on
`MachineExecutionWorkspace`. `ReadMachineExecutionTriage` returns a terminal
attempt's triage bundle over the authenticated machine seam.
