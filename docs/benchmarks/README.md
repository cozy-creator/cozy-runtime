# Public benchmark site

This directory contains the public MiniMax H3 comparison for GitHub Pages. The source repository stays private; the deployment publishes only the files explicitly selected by the benchmark site builder.

Public site: https://cozy-creator.github.io/cozy-runtime/

The site includes measured timings and sixteen original Turbo clips: two inputs, both engines, and one/four/eight RTX 5090s plus four H100 SXMs. Full regular-generation timings are included too.

Original videos are stored in a dedicated benchmark release asset rather than Git history. The builder verifies the archive and each media file against the checked-in manifest before assembling the site. No GPU inference is needed to build or publish it.

Use the `Publish benchmark site` workflow to deploy the checked-in site to GitHub Pages.

To build locally, download the asset named in `site/media-manifest.json` from its indicated release, then run:

```sh
python3 scripts/build-benchmark-site.py --media-archive /path/to/archive.tar.gz --output /path/to/empty/site
python3 -m http.server --bind 127.0.0.1 --directory /path/to/empty/site
```

The builder requires an empty output directory and checks every original media hash. Video bytes load only when a visitor starts playback. The public data file contains selected benchmark facts; private execution receipts and source files are not part of the publication.
