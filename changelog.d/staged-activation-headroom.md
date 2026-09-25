A successful component-staged request no longer permits its activation allowance
to disappear after a small change in available device memory. Restoring model
weights alone is not proof that the request fits beside them. The resident plan
waives an inapplicable activation bound only when the same request shape has
already completed without component movement; roomy devices can still satisfy
the measured allowance normally. Existing device-OOM retry fences remain unchanged.

Device entry now resolves the request before residency arbitration, so restoration
and capacity checks use its normalized shape. Requests known only to succeed with
component staging skip a full restore when their weights and activation allowance
cannot fit together, avoiding a refill that their plan would immediately undo.
