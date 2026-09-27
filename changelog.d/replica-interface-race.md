Concurrent replicas of one installation no longer refuse activation with
`package_interface_invalid: malformed_json`. Runtime parses the placement's interface
bytes in memory and publishes `package-interface.json` by atomic rename, so neither a
sibling replica nor an executor can read a partially written file.
