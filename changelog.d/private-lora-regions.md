LoRA-capable regions stay private in per-step and baked mode, preventing a refused
read-only export from dropping/refilling their bytes every stage. Already modified
or dirty bytes are also excluded from canonical layout sharing; plain regions retain
Degree2 attach/export behavior. CPU selection checks pass; GPU transition/output and
movement qualification remain required.
