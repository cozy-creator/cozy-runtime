# Public benchmark site

This directory contains the public MiniMax H3 comparison for GitHub Pages. The source repository stays private; the deployment publishes only the files explicitly selected by the benchmark site builder.

The site includes measured timings and sixteen original Turbo clips: two inputs, both engines, and one/four/eight RTX 5090s plus four H100 SXMs. Full regular-generation timings are included too.

Original videos are stored in a dedicated benchmark release asset rather than Git history. The builder verifies the archive and each media file against the checked-in manifest before assembling the site. No GPU inference is needed to build or publish it.

Use the `Publish benchmark site` workflow to deploy the checked-in site to GitHub Pages.
