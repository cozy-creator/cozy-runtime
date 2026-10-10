Published preparation still compared every controller-advertised image distribution/version and the exact host Python patch before selecting the package interpreter. Packages now install their complete locked closure in their own uv environment, so those unused image facts could reject a valid operation after an independent host SDK update.

Remove the inventory admission checks and unused internal parameters. The actual selected interpreter, exact requested release, locked wheel integrity, installed interface and model-binding contracts remain enforced.

The installed-Runtime observation bridge and image inventory reader also accept additive JSON metadata while retaining required fields, consumed types and ABI consistency.

Validation: eight focused tests pass. A real loopback wheel download and uv installation succeeds with stale, absent and empty inventory, imports the installed module, and still refuses a wrong application and wrong artifact hash. Focused lock/request/slot tests pass; Ruff passes excluding three pre-existing E501 lines in package_prepare.py. No native build or live SDK mutation.
