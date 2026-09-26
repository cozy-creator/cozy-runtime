Owner: /root
Purpose: preserve valid uv interpreter discovery when the caller works from home.
Branch: fix/uv-python-inventory-paths-20260926
Base: ec28ade0d1f93a3e754367bac1670766b8347916 (fresh origin/master)

Observed during the single-PRO H3 benchmark: uv emitted `.local/bin/python3.14`
from `/home/fidika`; Runtime rejected it before submission because probe requires
an absolute executable path. No active benchmark SDK or capture is changed here.
