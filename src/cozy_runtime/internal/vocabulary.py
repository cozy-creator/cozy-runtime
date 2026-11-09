"""The names a package may import from the Runtime's declaration modules, for the static
reader: each public module's exports, and every one of them as an attribute here. The reader
resolves a package's `from cozy_runtime.image import Steps` against this, importing nothing of
the package."""

from cozy_runtime import accel, author, fields, image, media
from cozy_runtime.accel import *  # noqa: F403
from cozy_runtime.accel.attention import AttentionLayout as AttentionLayout
from cozy_runtime.accel.attention import AttentionProducer as AttentionProducer
from cozy_runtime.accel.attention import ProducerCall as ProducerCall
from cozy_runtime.accel.attention import ProducerSite as ProducerSite
from cozy_runtime.accel.attention import attention_scope as attention_scope
from cozy_runtime.accel.attention import expect_attention as expect_attention
from cozy_runtime.accel.attention import producer_sites as producer_sites
from cozy_runtime.accel.attention import register_attention as register_attention
from cozy_runtime.author import *  # noqa: F403
from cozy_runtime.author._errors import AuthorError as AuthorError
from cozy_runtime.author._errors import ConformanceError as ConformanceError
from cozy_runtime.author._services import Secrets as Secrets
from cozy_runtime.author._services import Settings as Settings
from cozy_runtime.author._tool import Tool as Tool
from cozy_runtime.fields import *  # noqa: F403
from cozy_runtime.image import *  # noqa: F403
from cozy_runtime.media import *  # noqa: F403

_ATTENTION = (
    "AttentionLayout",
    "AttentionProducer",
    "ProducerCall",
    "ProducerSite",
    "attention_scope",
    "expect_attention",
    "producer_sites",
    "register_attention",
)
#: Each declaration module and the names it exports.
MODULES: dict[str, tuple[str, ...]] = {
    "cozy_runtime.author": tuple(author.__all__),
    "cozy_runtime.image": tuple(image.__all__),
    "cozy_runtime.fields": tuple(fields.__all__),
    "cozy_runtime.media": tuple(media.__all__),
    "cozy_runtime.accel": tuple(accel.__all__),
    "cozy_runtime.accel.attention": _ATTENTION,
}


def home(name: str) -> str | None:
    """The module that exports `name`, if one does."""
    return next((module for module, names in MODULES.items() if name in names), None)

