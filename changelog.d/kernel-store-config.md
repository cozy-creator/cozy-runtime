The machine kernel store reaches Sol-Attn through the executor's sealed configuration, not
environment reads in library code: the executor entrypoint configures it once, and the
background builder receives its namespaces as arguments. Fixes the env-authority fence that
blocked the 0.18.50 publication.
