A cut counts the step's own resident weights as warm, not cold: it runs finer before it evicts pages the step reads again (H3 15 s on a 4090 read 28 GiB more from the store on 0.34.11).
