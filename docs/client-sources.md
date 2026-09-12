# Sources in local scripts

Ordinary `async def main(ctx)` scripts can await three fixed native operations:

```python
from cozy_runtime.author.sources import (
    download_huggingface,
    download_civitai,
    convert_cozytensors,
)
```

`download_huggingface(repository, *, revision, profiles=(), carriers=())` requires
an immutable 40-character revision. One unambiguous carrier is selected automatically.
For a repository containing several models, select reviewed TensorFS profiles or explicit
carrier names. Native selection expands each selected index to its required shards.

`download_civitai(version, *, file="")` uses the provider's primary SafeTensors file
when unambiguous, or the explicitly selected file. TensorFS pins raw SHA-256 and exact
byte length; the provider's rounded size is not a byte identity.

Both return a `SourceArtifact`: an exact retained ordinary-file tree. Downloads finish
with the bytes verified and retained. Credentials come from the private deployment's
configured provider credentials; scripts do not pass tokens, delivery URLs, or Store paths.

`convert_cozytensors(source, *, profile)` converts one reviewed source profile and returns
a normal `ModelArtifact`. A family needing several profiles can convert them independently
and assemble the results through its model tools. The native converter owns source-header
validation, tensor transformations, and partial journals. Model results use a real native
metadata inheritance transaction and receipt without copying tensor payloads.

Each script starts from the beginning. A completed operation is skipped only after its
compatible result acquires independent recipient custody. Parameters, content, Runtime
implementation, native core and extension bytes participate in compatibility. Private
worker registry overrides also bind their exact bytes. Changing a parent script alone
can retain source/conversion results; changing the executing converter invalidates them.
Provider locations and credentials do not identify raw content.

Interrupted downloads retain a verified contiguous prefix. Fresh compatible calls can
independently adopt stopped progress. Conversion adoption snapshots already-fsynced native
tensor groups before continuing missing operations; the native snapshot never executes an
unfinished transform. Script stacks and arbitrary Python lines are not checkpointed.

Stopping or pausing execution retains partial work. Permanent cleanup follows the owner's
exact terminal acknowledgment with `retain_work=false`: native subprocesses finish stopping
before their source access and producer roots are released. A memo entry has its own root,
so cleanup of one script cannot remove another recipient's work. Process death also stops
the native subprocess; its durable native checkpoints remain resumable.

These source builtins are available to unpublished package jobs on local or rented machines
through the same Runtime/TensorFS code. They do not publish workflows or private memo rows
to Tensorhub. Dynamic provider resolution is refused inside a memoized package parent,
since otherwise the parent could skip resolution; conversion of an explicit SourceArtifact
remains a pure operation. Ordinary client `main` has no such enclosing memoization.
