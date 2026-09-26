Owner: /root/scheduler_finish (handoff from /root)
Purpose: private root acceptance priority and retained GPU groups, including singleton fanout.
Branch: fix/private-root-gpu-affinity-20260926
Base: f7d9ddaa1a449147788e64b528e58019aaba01b0 (review fetched origin/master d2ddc450).
PR: #649

Qualification is limited to focused private scheduler/prefetch tests. The global cozy CLI,
default ~/.cozy state, user rentals, and running workers are owned by the root task.
