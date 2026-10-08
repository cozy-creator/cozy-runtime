Model-bearing executor loads now require the machine's model-source grant before device,
package or measurement work. Installing a full TensorFS build no longer enables a fallback
to opening the model store or implicitly grants job source/output services. Weightless
loads and jobs that do not request storage still work without storage grants.

Standalone callers of the internal Checkpoint helper must explicitly use
`Checkpoint.from_store(store, manifest)`; managed checkpoints require a granted source.

Tracker: https://github.com/cozy-creator/tracker/issues/339.
