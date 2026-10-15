Runtime 0.24.0 bundles TensorD 0.5.30 and requires `tensorfs>=0.9.0,<0.10` (TensorFS native and Python share 0.9.0).

- TensorD 0.5.30: a run's measurements carry device memory and the machine keeps a `memory` log; an undeclared job takes the machine's GPU, else runs on CPU; a job reads its Hub model input in part, one closure per input; an update compares a named wheel by its bytes.
- TensorFS 0.9.0: a failed device fill refills instead of poisoning its region and `Closed` names its cause; part checkpoints; Hub control calls ride pooled connections.
- Minor: an undeclared job now means "wants a GPU" (was CPU), and `cozy_runtime.author.split` is new author API. Ship with Creator v0.6.35.
