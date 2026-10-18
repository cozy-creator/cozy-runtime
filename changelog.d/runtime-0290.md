Runtime 0.29.0 carries 0.28.2's fixes plus:
- An executor serves several calls at once (#1559). Calls take component scopes one at a time, first come first served. `Cancel` is a command, and TensorD 0.10's one-call-at-a-time driver and its cancel file still work. Batching across calls switches on with the TensorD that sends request-multiplexed frames.
- One decode at a time runs through the weight plane's paging hooks (#1607). The LM Engine pins the buffers its graphs read and records anew when one is rebuilt (#1604).
- A stage that runs once never evicts what a repeating stage keeps (#1589), and one thread maps weights back for the process (#1601).
It still requires TensorFS 0.11.
