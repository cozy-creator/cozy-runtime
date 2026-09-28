# Preparation failure diagnostics (h3a-092)

H3 request778 downloaded its base/PDD checkpoints and formed a four-GPU lane,
then failed in `Model.warm` with an NCCL error. The reply retained only the short
exception message. Its chained traceback disappeared when the executor was
retired, and the worker printed absent memory readings as peak0/allocated-1.
Those numbers were renderer defaults, not observations.

Preparation now retains a bounded structural traceback in the existing private
executor reply. The worker emits its lines through the existing boot activity
and captured log channels before retiring the executor. A reduced oversized
prepare reply preserves this diagnostic. No public configuration, database table,
protocol message or retry rule changes.

The trace contains at most four causal exceptions and eight innermost frames
per exception, bounded to40 lines of160 ASCII characters. It records exception
types and file basenames/function names/line numbers. It excludes exception
messages, notes, source lines, local variables and absolute directories. The
existing short refusal detail remains separate. These bounds deliberately favor
the innermost call site, without retaining live frame or tensor objects.

At the actual executor device boundary, every GPU's process also logs one startup record
before initializing CUDA. It includes declared rank/world, process credentials,
observed no_new_privs/dumpability when supported, and only CUDA_VISIBLE_DEVICES,
PYTORCH_ALLOC_CONF, PYTORCH_CUDA_ALLOC_CONF and NCCL_NVLS_ENABLE. Logging does not
import Torch or initialize CUDA; selected_device appears only when CUDA was
already initialized. This record uses captured stderr so it survives even when
initialization or a collective fails before an executor reply exists.

Missing, negative, non-integer and boolean memory readings render as unknown.
An explicitly reported integer zero remains a valid reading.

CPU regressions exercise a real chained `warm_generation` failure, retained
diagnostics through an oversized real IPC reply, secret/local/source exclusion,
bounds/cycles, unknown versus measured-zero memory, actual Linux protection
observation, and logging at the executor boundary without CUDA initialization.
Real multi-GPU warm qualification remains a separate gate.
