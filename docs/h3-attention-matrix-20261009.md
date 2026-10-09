# Explicit H3 attention precision matrix

Owner: Paul Fidika; implementation agent: linear_resume.
Branch: bench/h3-attention-matrix-20261009.
Base: origin/master c6fea990497a43d7f31bf016a8a06ae57cb14324.
Purpose: request-selectable full dense and four-dense/four-Sol attention comparisons on SM120, without changing defaults.

The running private cohort is allfast 6a429905054893bcac6514a7ba39fddc5f28f5a5d57cf885565c043392a709ba. Candidate wheels overlay only reviewed attention modules; existing bounded linear GEMM, LoRA bake, fusion, native executable and dependencies are preserved. Root owns GPU qualification and installation.

The existing FP8 Sol selectors use Sage or NVFP4 dense calls and do not represent pure FP8 dense attention. This task adds explicit dense FP8/F16 and BF16/F32 controls, with matching mixed selectors. Full-dense controls must never invoke sparse routing. All numerical changes remain opt-in pending the user's video review.
