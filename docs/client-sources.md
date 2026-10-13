# Provider sources

A run's code does not fetch Hugging Face or Civitai sources: its executor refuses to dial (cr-012).
A machine fetches, verifies, converts and keeps a source itself, so it is resumable, shared by
every run on that machine, and normalized at ingest:

- As a model input: `cozy run <org/package/function> model.<slot>=hf://<org>/<repo>@<revision>`
  (or `civitai://<version>`), with `--source-profile <slot>=<profile>` to pick the layout, on a
  named machine (`--rental=<name>`). The function receives a `Model`.
- As a checkpoint: `cozy model upload hf://<org>/<repo>@<revision> <org/name> --rental=<name>`
  makes it on that machine and uploads it; later lanes are `cozy run <conversion function> …`
  on the same machine.

`cozy_runtime.author.sources` (`download_huggingface`, `download_civitai`, `source_files`,
`convert_cozytensors`) runs on no current machine; a call is refused naming these two routes.
