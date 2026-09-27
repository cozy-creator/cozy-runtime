Request shape cells now see what sizes H3's memory. An `Assets` parameter's occurrences count as
the `assets` axis (they arrive as `AssetInputValue` rows, which the count missed), and a `Shape`
table may cover an int bounded by `msgspec.Meta(ge=..., le=...)`, so H3's `duration_s` maps to
`frames` without changing its wire type. A first 15 s request after 10 s ones is then an
unmeasured cell and staged admission frees the card for it, instead of reusing a 10 s peak.
