# Idle LM owner retirement

The LM engine's idle shrink callback previously retired its decode graphs and KV buffers
but retained the graph weight Bindings in `addresses`. Those Bindings kept weight Pins
alive until explicit model unload, even after no cached executable could use them.

The callback now claims the driver plane and condition without waiting, snapshots the
actual Binding inventory, and tries its Domain, Binding and Memory locks. Active rows,
pending work, lost completion or incomplete events decline shrink. Completed graphs are
reset before their Bindings close; clearing `addresses` and `where` makes the next ordinary
generation acquire fresh bindings. No byte sizing or allocator policy is added.

The real CPU proof writes a seeded small Qwen3ForCausalLM to a TensorFS checkpoint and
loads it through the actual resolver, Backend, ModelRegistry and Placement. Public
generation progress observes an occupied row and one Region pin; discard refuses. On the
old source, idle discard removes KV but leaves one Binding and two retained blocks. The
candidate closes the Binding and leaves zero retained blocks. First and rebuilt generation
match ordinary Transformers exactly: `[41, 54, 12, 16, 41, 13, 10, 14]`. CUDA is uninitialized.

The separate GPU fixture requires actual graph capture and replay, native active-owner
refusal and idle retirement, escaped output/KV storage integrity, rebuild, explicit close
and a matched working-memory baseline. Its random-weight model proves the lifecycle;
it does not measure pretrained-model quality, live sequence preemption or paged KV holes.
GPU qualification is pending. DerivedCache74cce and CUDA recovery8384 remain unchanged.
