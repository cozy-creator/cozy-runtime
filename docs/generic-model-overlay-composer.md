# Generic model-slot LoRA overrides

A request can add an ordered LoRA stack to a normal model selection. The authored
function retains its usual Model parameter; it needs no LoRA argument. A model
library opts into the components it can adapt through `__adapter_compatibility__`.
The shared Runtime composer supports ordinary linear A/B factors and refuses
unsupported keys, shapes, target types or model components with an adapter error.
Generic model packages declare `cozy-runtime[adapters]` to install the PEFT backend;
`model-execution` remains TensorFS-only.

Model choices travel with a release root, a captured execution, or an authenticated
private placement. A bare parameter selects the root's own model. A qualified
`motion_segment_turbo.models.base_model` choice selects that captured H3 child;
`paul/qwen-image-2/generate_image.models.model` names an external captured callable.
Runtime validates the exact installation, callable and parameter before resolving
weights. Two children with the same parameter name do not share an override.
Private placement choices always name the full callable model slot.

Runtime resolves checkpoints and provider sources through its existing catalog and
native source services. Provider credentials remain in memory and native-child
stdin. The preparation uses a native CPU child to build an immutable TensorFS
component view: base weights and every encoded physical part are grafted unchanged,
and ordered LoRA factors become ordinary managed destinations. It requests no GPU
lease for source acquisition or this derivation.

The package's own SDK constructs upstream PEFT layers before the ordinary census,
fill, optimization and inference path. Alpha/rank and each requested strength apply
once; changed order or strength produces a distinct model construction. Factors
participate in the existing MemoryManager residency and gang admission. A requested
unsupported stack fails the request rather than serving the pristine base. Old
SDKs must advertise `peft_graph/1` for adapted construction/loading, and old Runtimes
refuse model overrides independently of ordinary base-only work.

H3 opts into `fl2va_dit` and `ref2va_dit`. Its built-in Turbo slot remains separately
owned; generic PEFT wraps those projections while retaining the Turbo residual and
selectable output heads. Toy generic-model and actual tiny H3 CPU forwards verify
ordered arithmetic, pristine base values, repeatability and Turbo cleanup. A native
encoded-leaf CPU proof verifies physical parts and row scales. These checks do not
establish GPU staging, distributed H3 inference, or the quality of an arbitrary
third-party face adapter.
