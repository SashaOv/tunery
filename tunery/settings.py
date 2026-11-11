"""Render settings: one model serves as the `render` flags and the config keys.

A config key is the long flag without `--` (`concert-sub: false` is
`--no-concert-sub`). Layers apply in order, each changing only the keys it
states: defaults, the project configuration, the layout's `config:`, the
command line.
"""

from collections.abc import Iterable
from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=lambda name: name.replace("_", "-"),
        populate_by_name=True,
    )

    concert_sub: Annotated[
        bool, Parameter(alias="-cs", negative=["--no-concert-sub", "-nocs"])
    ] = Field(
        default=True,
        description="Let transposing parts (Bb, Eb, F) fall back to concert charts; "
        "--no-concert-sub stops at their own charts.",
    )
    current_dir: bool = Field(
        default=True, description="Search the layout's own folder, first."
    )
    explicit_bookmark_hierarchy: bool = Field(
        default=False,
        description='Prefix part bookmarks with "_ " for readers that show one '
        "bookmark level.",
    )

    def layered(self, layers: Iterable["Settings"]) -> "Settings":
        """Apply `layers` in order, each overriding only the keys it set."""
        result = self
        for layer in layers:
            result = result.model_copy(update=layer.model_dump(exclude_unset=True))
        return result
