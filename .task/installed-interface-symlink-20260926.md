Owner: /root/h3_cozy_astra
Purpose: let published package preparation read uv-linked installed interface metadata while retaining strict artifact-cache and receipt readers.
Branch: fix/installed-interface-symlink-20260926
Base: f801af82f72622e0d4deb184417e2c81da94e6ce (origin/master fetched 2026-09-26)

Evidence: owned H100 rental Titus refused published minimax-h3 1.17.1 before inference because package-interface.json is a legitimate uv symlink into the immutable image dependency seed. The resolved regular file is 29,893 bytes, SHA256 af45a4609af5d2ab1fc943dcb5a3cdb7e01bc920754c72985ef081fe9f677102.

Validation: source review and necessary serial native-donor wheel build only. User prohibits tests, CI, lint and vet. Root owns merge and ordinary CLI updates of the owned rentals; no pod source edits or global SDK changes by this task.
