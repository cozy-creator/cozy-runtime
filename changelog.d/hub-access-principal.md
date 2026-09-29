### Fixed
- Keep each machine's Hub access bound to its original delegated account. Same-account
  refresh remains available; another account is refused before replacing valid access.
- Refuse changed delegated principals across Runtime catalog, TensorFS, and publication
  access instead of silently using another account's credentials.
