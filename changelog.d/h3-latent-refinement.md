Retain completed H3 latents, reference conditioning and original PCM in a bounded
artifact for learned spatial upscaling. Refine video on a suffix of the standard
30-step schedule with audio locked, then fit decoded chunks to an exact output
canvas while preserving aspect ratio. The original generation remains reusable.
