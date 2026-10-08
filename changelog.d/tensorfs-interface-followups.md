Runtime 0.21.1 bundles TensorD 0.3.1 and requires TensorFS >=0.6.1,<0.7.
Model prefetch now honors explicit model choices on the Runtime/TensorD control path.
Weight recovery and fixture annotations pass the full configured type check, and release
publication explicitly selects source builds or the verified complete prebuilt file set.
Installed managed-call proxies retain their public callable identity for prefetch.

TensorFS 0.6.1 preserves inherited companion files, including tokenizer assets, by
reference without copying their payloads. The dependency floor selects the fixed local library;
machine/executor peers still use operation and capability contracts without requiring
matching versions.

Bundled TensorD also honors declared accelerator jobs and their managed descendants,
scopes local package reuse to immutable SDK and client selections, and observes finite
cgroup-v1 memory pressure. Same-filename updates select a fresh package generation while
earlier accepted generations retain their original inputs.

Delivery: https://github.com/cozy-creator/tracker/issues/350
Changes: https://github.com/cozy-creator/tracker/issues/345,
https://github.com/cozy-creator/tracker/issues/347,
https://github.com/cozy-creator/tracker/issues/348,
https://github.com/cozy-creator/tracker/issues/351,
https://github.com/cozy-creator/tracker/issues/352,
https://github.com/cozy-creator/tracker/issues/353
