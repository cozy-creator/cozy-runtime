An attempt's metrics carry torch's reserved peak (`peak_reserved_bytes`) beside its allocated one;
TensorD records both in the run's `memory` measurements, which `cozy run show` prints.
