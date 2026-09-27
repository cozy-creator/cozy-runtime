A sequence-parallel group whose rank 0 cannot hold every component keeps a `@placeable`
one resident on a follower instead of re-staging it on rank 0 for every request. Rank 0 and
the other followers free it; every module call into it (root or submodule) runs on its home
rank under the leader's scope and returns tensors or a transformers model output through the
attempt spool; an out-of-memory in such a call sheds the rank's other components and runs it
once more. Every rank derives the same placement from the agreed ceiling and component sizes.
MiniMax H3 declares its 51.5 GB text conditioner placeable: on 4xH100 it lives on rank 1
beside that rank's DiT shard, which removes rank 0's 8.9 s per-request text-encoder stage.
The conditioner now computes only the hidden state H3 reads (`last_hidden_state`, the same
tensor as `hidden_states[50]`), and releases Qwen's position cache from a forward hook.
