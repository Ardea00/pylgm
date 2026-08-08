"""Link functions for likelihoods."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityLink:
    """The identity link."""

    name: str = "identity"
