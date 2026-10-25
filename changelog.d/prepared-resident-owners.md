Prepared GPU repeats accept placed component roots as owners. Resident-only preparation refuses
missing weights without a source read, retains fixed fill and kernel identities, restores capture
state, and keeps the surrounding request's later component phase priorities intact.
