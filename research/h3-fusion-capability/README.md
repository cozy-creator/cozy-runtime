MiniMax H3 fusion compatibility investigation
===========================================

The captured Diffusers 0.41 SwiGLU return annotation changes the whole-function source digest, preventing all H3 glue fusion under `fusion="accept"`. This task replaces source fingerprints with the callable and model topology contract consumed by the substitution. Numerical checks must cover actual Diffusers 0.40 and the captured 0.41 dependency. GPU admission and full inference remain pending. No attention or precision default changes are included.
