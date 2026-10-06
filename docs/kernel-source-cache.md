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

Tracker: https://github.com/cozy-creator/tracker/issues/334
