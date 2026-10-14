"""A test script body declared as a CPU job (`[tool.cozy] accelerator = false`)."""

DECLARATION = "# [tool.cozy]\n# accelerator = false\n"


def cpu(body: str) -> str:
    if body.startswith("# /// script\n"):
        return body.replace("# /// script\n", "# /// script\n" + DECLARATION, 1)
    return "# /// script\n" + DECLARATION + "# ///\n" + body
