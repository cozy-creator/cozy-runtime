`author.concurrently(*calls)` runs independent calls and returns their results in order. A
public Model method whose declared components the group keeps on other ranks (`placeable`)
holds nothing on this device: its scope is remote, admitted on the home rank, and runs on its
own thread beside the one local scope. Everything else runs here in order, so without a
hosted component it is the plain sequence. MiniMax H3's `condition_references` uses it: rank 1
runs the text encoder while rank 0 encodes the references, and the state equals the
sequential one bitwise.

`author.spread` offers each call to every idle follower. A follower that cannot take it on
measured room declines and the call runs elsewhere; a follower homing a placeable component
always spares (never evicts), so rank 1 decodes H3 clips only beside its resident text encoder
and DiT shard. `spread(..., spare=True)` spares on every follower. Calls in flight to one
follower never interleave; the attempt's execution record lists each spread's calls per rank
and declines.

H3 reference latents are memoized per generation (`loader.cache`, now least-recently-used at
its bound) by the exact pixels, construction and encode parameters; audio is encoded every
time. Image references that miss spread over the idle ranks.
