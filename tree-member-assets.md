# Native Tree member assets

Implementation contract: Tracker cr179. Base public Runtime0.18.1 plus the shared
wire56 Model-default prerequisite. Package assembly and prefix formats remain
package-owned.

`tree.member(path, ImageAsset, bound=AssetBound(...))` projects one regular member
from a verified held native Tree into the existing typed Asset surface. Runtime
checks canonical membership, path, digest/length, media and byte bounds; author
code never supplies a local hydrated path. MediaDecoder and normal child calls
consume the resulting attempt-scoped read capability. Output copy/re-export uses
the existing Outputs writer and native output commit.

The implementation also propagates an explicitly declared fixed-output aggregate
budget through existing worker/executor admission and output checks for a final
video plus prefix Tree. Default budgets and separately bounded root input intake
remain unchanged. No new storage authority, file manager or general workflow DSL.
