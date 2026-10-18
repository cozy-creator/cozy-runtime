Runtime 0.28.2 replaces 0.28.1, which failed its gate and was never published. It fixes 0.28's memory regressions on 8 and 12 GB cards:
- An op that runs out of memory never unmaps what the component that ran last holds (#1608; 12 GB H3's dead-pointer Triton launches).
- Bands move other tensors off before a result goes to host memory (#1605; 8 GB H3).
- A follower GPU's cap counts what it holds, and a failed attempt gives back what it took (#1598; 2× 8 GB H3).
- Measuring a call never changes model state (#1602, #1591 for 0.28; SDXL HiDiffusion).
