# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.12.0,<1"]
# ///
"""Run with `cozy run ./examples/client-scripts/source_files.py`.

Add `--rental-only` to execute on a private rental. Retained download work is
reusable when this script is edited or run again on the same machine.
"""

from cozy_runtime.author import ScriptContext
from cozy_runtime.author.sources import download_huggingface, source_files

REPOSITORY = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
FILES = ("config.json", "preprocessor_config.json")


async def main(ctx: ScriptContext) -> None:
    source = await download_huggingface(REPOSITORY, revision=REVISION, files=FILES)
    tree = await source_files(source)
    ctx.log(f"Verified {tree.size_bytes} bytes in {tree.digest}")
    for path in tree.files():
        ctx.log(f"{path.relative_to(tree.path)}: {path.stat().st_size} bytes")
