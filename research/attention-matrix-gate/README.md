# Attention arithmetic gate

Run through ordinary `cozy run ./gate/verify --rental=<owned-rental>` after the root installs the matrix Runtime wheel. This is a tiny arithmetic/dispatch check; it does not claim full-video quality or speed.

Before describing/capturing this research fixture, materialize `vendor/diffusers-0.41.0.dev0-py3-none-any.whl` from the exact Diffusers source commit `578c9b2c6636ab2424a0e56186268b83623656b2`. The prepared artifact is SHA256 `6b55b6d7b4c6d2249729391d74aec6a73e57e90d2efe7eabf7fbd93268e4317e`; its 1041 package payload files and non-RECORD metadata match the installed benchmark environment. The wheel is an input artifact, not repository source. Add the intended Runtime candidate as the other local uv source. Normal CLI capture relocates both wheel paths and sends them with this code.

Run5022 failed before GPU execution because the worker has no Git executable, while the prior fixture asked uv to fetch Diffusers from Git. This version captures the wheel, preserving that exact source without requiring a VCS checkout on the worker. No remote installation or system changes are made by the fixture author.
