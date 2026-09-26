Owner: /root/h3_cozy_astra
Purpose: restore the placement-owned development mapping used by Acquirer._bindings after the wire-62 installation refactor.
Branch: fix/placement-development-binding-20260926
Base: 5aec87d71bea4c3ff65267afaa01ab4ed2366c70 (origin/master fetched 2026-09-26; includes PR627 and PR628)

Evidence: ordinary published H3 run1079 on owned Titus reached prepared placement, then worker convergence failed before any GPU execution with NameError: name 'development' is not defined. The package label and empty-entrypoint validation still reference that missing local value. This area was not changed by PR628's installed-interface reader repair.

Validation: independent source review and one necessary serial native-donor wheel build. No tests, CI, lint or vet under user instruction. Root owns merge and CLI updates; all remote evidence and default package outputs are preserved.
