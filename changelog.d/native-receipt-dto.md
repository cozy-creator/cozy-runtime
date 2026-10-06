Native executor weights storage no longer imports the retired worker protobuf
bindings. Execution receipts use Runtime-owned DTOs and canonical JSON while
retaining the stored receipt bytes, digest, default omission and native size bound.
Importing Runtime no longer loads the old generated bindings, and its version
command no longer advertises the retired worker wire contract.
