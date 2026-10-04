A GPU keeps the display floor only when a display is active on it (NVML's display active state).
A connector with a display attached, as RunPod's L4 and A40 report, no longer counts: those cards
kept 1/16 of their memory free (1.5 GB on an L4), and a full card left a stage no budget.
