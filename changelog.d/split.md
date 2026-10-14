`cozy_runtime.author.split(fn, x, *cuts)`: a package declares how an op may be cut (`Cut`: dim, scale,
overlap, align) and the Runtime sizes the pieces from the op's measured bytes per input element and the
memory free (idle weights unmap first); a piece that runs out runs again at half the size. SeedVR2's VAE
tiles and H3's tile batch move onto it.
