"""Link functions for likelihoods."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IdentityLink:
    """The identity link."""

    name: str = field(default="identity", init=False)
