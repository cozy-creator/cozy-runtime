# Kernel sources with bare worker images

Runtime's existing PINS and recipes are the default source authority. Constructing a kernel
job computes its same source/compiler/device key without reading an image index or downloading.
An already compiled artifact therefore remains usable offline.

Only a builder that actually opens a missing source acquires it. Store archives beneath the
sealed COZY_KERNEL_CACHE namespace, serialize each digest with a file lock, and use the existing
Runtime egress/storage admission and atomic digest verification. Reuse a read-only preseeded
archive only when its bytes match the Runtime pin; its index cannot override the Runtime.
Setup builds fetch only their source/place archives and build dependencies absent from the
environment. The Python recipe keeps its existing aggregate Python-source inputs. Do not
download other setup sources, change recipes/pins or introduce image qualification.

An explicit registry root remains a component-test/custom-source input. It does not replace
the normal Runtime pins. Failures remain visible compilation failures and never publish a
partial archive or artifact. No numerical/kernel-selection changes or local GPU proof.

CPU proof uses tiny HTTP-served wheel and tar sources with the actual builder subprocess and
an import from its published artifact. It covers concurrent source acquisition, corrupt-cache
repair, matching preseed reuse, checksum refusal, private redirects and the existing storage
owner's refusal. It also removes downloaded archives, stops HTTP and reads the compiled
artifact without starting another builder. Equivalent explicit-registry and pin inputs keep
the same artifact key.

The acquisition calls the existing Runtime egress helper, including its storage-admission
hook. This does not add a host-capacity owner or propagate retired supervisor channels to
builder subprocesses; the existing no-channel local-write behavior remains unchanged.

Tracker: https://github.com/cozy-creator/tracker/issues/334
