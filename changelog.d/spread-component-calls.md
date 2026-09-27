`author.spread(component, method, arguments)` runs a component's pure method once per
argument tuple, in order. In a sequence-parallel group the runtime runs each round of calls on
rank 0 and every follower that hosts no placeable component, returning each follower's tensor
straight to rank 0 over the group (header then data) and carrying the caller's autocast;
elsewhere the calls run locally. MiniMax H3's video VAE decodes its temporal clips this way:
each clip decodes from its own latent window, and rank 0 cross-fades them in order, so the
frames are what one rank computes on identical cards. On 4xH100 with the text encoder hosted
on rank 1, 14 clips decode in 5 rounds on ranks 0, 2 and 3 instead of 14 on rank 0.
