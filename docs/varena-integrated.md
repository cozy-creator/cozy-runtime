# Varena v2 Runtime integration

Owner: `/root`, xs-138. Branch `codex/xs138-runtime-integrated` starts from fetched
master 640193706bf535a99fe167b363c6aea28972cf52 and combines:

- native Heads/INT8, projection, Tail and phase-region Plan: 3daaaeda979ca3da4b21647893dfcfa508e6cf29;
- native cooperation startup, boundaries and error cleanup: 4d5844f0ffb0af23dbb4f19a9e143c1d179203d6.

The component boundary now wakes the native participant and preserves the declared
phase-region inventory. Operator implementations, pinned kernel identities and
allocation menus are unchanged from the qualified INT8 candidate. Unchecked
cleanup still passes its unchecked mode through to owner wake, preserving the
original operation error.

Source 3d5caf977 has zero additional diagnostics against the frozen INT8 parent in
matched checks: Ruff 12/12 and mypy 47/47. The existing findings are not a full
lint/type pass. GPU validation must use one coherent native core and this Runtime;
the earlier independently qualified model/pressure cohorts do not prove their
combination. Source and output artifacts of the full 25324/25347 comparison remain
unchanged.

Cache-owner callbacks, foreign-pressure policy, remaining sparse/closed execution
and the public shared-core ABI remain separate work. No release is authorized by
this integration checkpoint.
