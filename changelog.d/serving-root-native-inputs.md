A rented or local inference execution with file inputs no longer refuses at grant with
`grant_native_unavailable`. Runtime adopts only a job root's native inputs at submission; a
serving root now dispatches from the submitter's own hold instead of an execution hold that was
never created. The refusal now also names the underlying cause.
