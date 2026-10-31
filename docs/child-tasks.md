# Child tasks

`app.child_task` registers a function only its own package's code calls: a managed child of the
same installed package revision, never a root request or another package's call. It keeps its
executable metadata and model bindings and takes `app.entrypoint`'s options except `hidden` and
`model_classes`.

```python
app.child_task(regenerate_shot)


@app.child_task(accelerator=False)
def plan_pieces(payload: PlanRequest) -> PlanOutput: ...
```

The package interface keeps a child task with `internal: true`. Public discovery and interface
wheels omit it; installation and same-package capture keep it. `cozy run <package>/<child task>`
is refused. TensorD enforces this from admitted capture and child journal identities; a
caller-supplied name or a parent request ID confers no authority.
