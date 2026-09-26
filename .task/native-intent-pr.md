Ordinary run1123 reached the native source facade but never completed: native admission reconstructed an obsolete interface-bearing call digest, recorded `child_admission_refused`, and the native result branch ignored that durable refusal while waiting for a native row that had never been created.

Dispatch, the managed-call journal, and native-source admission now use one canonical identity builder for module/export/request and optional capture. The existing hash comparison, source-operation mapping, parent-attempt fence and stable service identity remain enforced. Machine call results return a recorded refusal before polling the native service, so an awaited rejected source call terminates its parent.

The real Worker fixture now uses current installation IDs, transferred files and a fully locked Runtime dependency closure. It qualifies both native HTTP download→source view→conversion→managed Model consumption and a conflicting frozen native intent whose actual parent must fail. Related source memo/custody and commit-file fixtures use the current wire fields.

Validation: both real Worker/native integration cases passed (16.6s); 46 source memo/custody/file regressions passed, with six existing skips. Ruff and focused mypy passed on the changed production files (generated worker_pb2 diagnostics excluded only in external qualification config). No global worker or captured environment was patched. Normal installed-CLI replay on the rebuilt SDK remains the deployment gate.

Evidence: `outputs/qualification-memo-20260926/native-intent-integration.log`, `native-custody-regressions.log`, and live run1123 history (canceled after the durable refusal proved it could not progress).
