Build the local quality-review index without modifying the run manifest:

```sh
python3 research/h3-quality-matrix/review_matrix.py MANIFEST.json --out REVIEW_DIRECTORY
```

The manifest contains `runs` with `id`, `kernel`, `schedule`, `scene`, `seed`,
`video`, `run_show`, `status` and `timing_note`. Optional `kernel_label`,
`arithmetic_note` and `attention_selector` make the displayed precision and
observed selector explicit. Root owns the manifest; this helper writes only its
separate review directory.

Completed clips receive a full software AV decode, SHA256, geometry, 362-frame,
24fps, 32kHz stereo and duration checks. The run receipt must establish eight
denoising steps and execution time. On this one-GPU H3 matrix, `dense4-sol4`
requires 192 sparse calls and `dense8` requires zero. Pending/failed/missing
artifacts remain visible and do not count as verified. Existing matching AV
receipts are reused; original videos are hard-linked or copied into `media`.

`index.html` is static and needs no server, framework, network fonts or autoplay.
`index.json` retains all run metadata and check results. AV verification does not
approve visual quality, audio quality, or defaults; the user reviews those.
