# H3 request profiling

This research package delegates to MiniMax H3's registered render handlers. It records a full CUDA trace, phase ranges and weight-plane counters while retaining the original video. It is not part of the production Runtime package or its attention defaults.

Version 0.0.12 was exercised through ordinary Cozy requests 5089 and 5090 with profiling disabled, then 5091 with a full CUDA capture on Runtime 0.21.7 and TensorFS 0.6.1. All eight steps and the original audiovisual output were retained. Recorder and entrypoint source are preserved from that qualified snapshot.

The common `trace_summary.py` parser reads ChromeTrace JSON or gzip incrementally. `research/h3-single5090/attribute_trace.py` uses it to associate GPU activities with their launching CPU ranges. Profiled sums are not clean inference timings; independent streams and nested phases overlap.
