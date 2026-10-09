### Removed
- The Runtime holds no Hub authority and has no Hub path (th-241): the unused host-handoff reader
  (`read_worker_host_config`, which read a worker token from `machine-publication-authority.json`),
  the uncalled output upload `egress.put_from` with `TENSORHUB_OBJECT_STORAGE_HOSTS`, and the
  account-index rewrite to "this machine's Hub" in package installs are gone. Hub access is TensorD's.
