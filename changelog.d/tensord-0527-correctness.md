Runtime 0.23.34 bundles TensorD 0.5.27.

- Accepted remote editable runs retain the source generation they were submitted with; direct local paths keep their intentional live behavior.
- Source recovery and collection preserve named environments when source inventory is unavailable.
- Successful attempt receipts keep actual executor Runtime/TensorFS and TensorD build identity.
- Linux TensorD processes keep the `tensord` name through update and proc-executable launches.

Runtime Python/native execution inputs are unchanged from 0.23.33. The native wheel reuses the verified public 0.23.33 donor. Qualification for the changed controller paths uses actual CPU process/socket/storage checks; this release adds no new GPU inference qualification.
