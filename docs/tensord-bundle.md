# Tensord executable ownership

The Linux Runtime wheel installs `tensord` from `.data/scripts/tensord`.
`task tensord` builds the source in `tensord.pin` from cozy-creator/tensord.
Standard wheel RECORDs, ELF/mode/platform checks, actual install/version identity
and uninstall verification still apply. This build pin identifies the owned artifact;
it does not gate client/server operation on source equality.

The Runtime API, native extension, declared package dependencies and embedded
installer library are unchanged. Existing private repository read-secret identity
remains configured while its fetch URL uses the renamed repository.

Initial Docker provisioning bakes this wheel and its declared dependencies and
runs tensord directly; the boot path does not download Python software.
Machine and controller rename PRs accompany this artifact path change.
The initial draft pin names the reviewed-source rename checkpoint; before readiness
it will be updated to the merged machine rename source.

Tracker: https://github.com/cozy-creator/tracker/issues/336.
