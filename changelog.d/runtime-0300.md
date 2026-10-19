Runtime 0.30.0 ships as one cohort with tensorfs 0.12.0, cozy v0.11.0 and Tensorhub's v2 release. It carries 0.28.1, 0.28.2 and 0.29.0, none of which was published. It requires tensorfs 0.12 and the cohort's TensorD.
- TensorFS v2 (#1600): weights decode by scheme name from cozytensors/1 checkpoints. A model loaded through TensorD gets every file its manifest lists (configs, tokenizers, assets). Derivations run on TensorD's derive channel.
- Memory in varena (#1617, #1618, #1619): one component decides every GPU byte. Weights live in varena's memory, the executor speaks TensorD's ledger wire (a hard cut from v0.10), and attention and splits cut through varena's memory; bands and spill are gone. A package may declare its model sequence (`ctx.plan`), and varena evicts and restores by it.
- D10: FILL AT CUT (component-model header).
- An executor serves several calls at once (#1559). One decode at a time runs through the paging hooks (#1607), and the LM engine re-records when a buffer its graphs read is rebuilt (#1604).
- Measuring a call never changes model state (#1602).
- No local paths in the repo, with a fence (#1621); ci.yaml runs on every PR and master push (#1625).
