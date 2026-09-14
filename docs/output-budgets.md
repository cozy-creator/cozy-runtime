# Fixed byte-output budgets

The accepted invocation grants each fixed byte output separately. Runtime adds
their admitted capacities for the attempt's aggregate output budget and sends the
result to the executor. Two 256 MiB outputs therefore admit 512 MiB combined.
Each individual output keeps its own limit, at most the existing 256 MiB ceiling.
Invocations without byte-output declarations retain the 256 MiB default. Model
manifest outputs keep their separate TensorFS weights capacity contract.

Encoding and final publication validate the same aggregate again. The executor's
reply can use a smaller budget but cannot raise the grant. Native `Outputs.commit`
reads the recorded parent invocation instead of taking a budget from the script.
Its authored pending-file commits share that budget. Selecting a native Tree
member is an input capability and does not charge a second spool copy.

Received child-result bytes use the parent's declared aggregate, cumulatively
over the attempt. Root input intake keeps its separate existing 256 MiB limit.
Native storage admission still reserves capacity before writes.

`Annotated[Tree, AssetBound(max_bytes=...)]` and its optional form have identical
static and imported interfaces. Tree bounds cover total member bytes. Media and
decoded-byte bounds belong on `tree.member(...)`, which checks each projected
typed asset before `MediaDecoder` or child forwarding can consume it.
