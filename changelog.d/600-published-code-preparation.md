Published serving-only dependencies can now retain their exact installed code and
interface before a caller selects models. This enables a published CPU parent to
capture a modeled serving child without eagerly downloading its default model.
The code-only state has zero executable bindings. Selected-model and missing
weightless-binding checks remain enforced, as does exact callable inventory
verification at execution admission.

A real Worker regression installs and describes a serving-only package, reopens
its sealed receipt, and acquires zero bindings without invoking its model loader
or handler. Both original refusals reproduce on0.18.24. No model numerical code,
protocol fields, or native kernel inputs changed.
