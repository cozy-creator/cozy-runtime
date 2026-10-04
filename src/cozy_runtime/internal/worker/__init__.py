"""What is left of the Python worker: the import closure of the weights writer
(`weights.py`, `weights_finalize.py`), held until the Rust machine writes weights outputs.
Nothing starts a worker any more; these modules go with the writer."""
