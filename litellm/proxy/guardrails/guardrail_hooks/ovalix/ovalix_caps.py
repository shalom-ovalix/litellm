"""Per-call inspection caps the Ovalix tracker serves, and which items of a call they leave inspected."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate, chain, islice, takewhile
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

CapHandling: TypeAlias = Literal["block", "trim", "allow", "check_up_to_cap"]


class ItemKind(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FILE = "file"


class CapLimit(str, Enum):
    MESSAGES_PER_CALL = "messages_per_call"
    FILES_PER_CALL = "files_per_call"
    TOOL_CALLS_PER_CALL = "tool_calls_per_call"
    TOOL_RESULTS_PER_CALL = "tool_results_per_call"
    CALL_SIZE = "call_size"


class OvalixCaps(BaseModel):
    """The org's caps as the tracker's caps endpoint returns them."""

    model_config = ConfigDict(frozen=True)

    message_chars: int = Field(ge=1)
    messages_per_call: int = Field(ge=1)
    files_per_call: int = Field(ge=1)
    tool_calls_per_call: int = Field(ge=1)
    tool_results_per_call: int = Field(ge=1)
    call_size_bytes: int = Field(ge=1)
    file_size_bytes: int = Field(ge=1)
    hit_behavior: CapHandling


class ItemRef(NamedTuple):
    kind: ItemKind
    position: int


@dataclass(frozen=True, slots=True)
class CapPlan:
    """Which items of a call are inspected, and the cap the newest item reports when one was hit."""

    kept: frozenset[ItemRef]
    newest: ItemRef | None = None
    cap_hit: CapLimit | None = None
    handling: CapHandling | None = None

    @property
    def newest_only(self) -> bool:
        return self.handling in ("block", "allow")


_COUNT_CAPS: Final = (
    (CapLimit.MESSAGES_PER_CALL, ItemKind.MESSAGE),
    (CapLimit.FILES_PER_CALL, ItemKind.FILE),
    (CapLimit.TOOL_CALLS_PER_CALL, ItemKind.TOOL_CALL),
    (CapLimit.TOOL_RESULTS_PER_CALL, ItemKind.TOOL_RESULT),
)
_SIZE_BUDGET_ORDER: Final = (ItemKind.MESSAGE, ItemKind.TOOL_RESULT, ItemKind.TOOL_CALL, ItemKind.FILE)
# LiteLLM takes rewritten text back only through `texts`, which holds messages; dropped items of any other
# kind would reach the model untouched, so a trim that drops one becomes a block.
_TRIMMABLE_KINDS: Final = frozenset({ItemKind.MESSAGE})


def _count_limits(caps: OvalixCaps) -> Mapping[ItemKind, int]:
    return MappingProxyType(
        {
            ItemKind.MESSAGE: caps.messages_per_call,
            ItemKind.FILE: caps.files_per_call,
            ItemKind.TOOL_CALL: caps.tool_calls_per_call,
            ItemKind.TOOL_RESULT: caps.tool_results_per_call,
        }
    )


def _refs_of_kind(sizes: Mapping[ItemKind, Sequence[int]], kind: ItemKind) -> Iterator[ItemRef]:
    return (ItemRef(kind, position) for position in range(len(sizes.get(kind, ()))))


def _newest(sizes: Mapping[ItemKind, Sequence[int]], first_over_count: ItemKind | None) -> ItemRef:
    candidates: Final = (ItemKind.MESSAGE, first_over_count) if first_over_count else _SIZE_BUDGET_ORDER
    kind: Final = next(kind for kind in candidates if sizes.get(kind))
    return ItemRef(kind, len(sizes[kind]) - 1)


def _newest_first_within_count(
    sizes: Mapping[ItemKind, Sequence[int]], limits: Mapping[ItemKind, int], kind: ItemKind
) -> Iterator[ItemRef]:
    count: Final = len(sizes.get(kind, ()))
    return (ItemRef(kind, position) for position in range(count - 1, count - min(count, limits[kind]) - 1, -1))


def _kept_within_size(
    sizes: Mapping[ItemKind, Sequence[int]], limits: Mapping[ItemKind, int], newest: ItemRef, budget: int
) -> frozenset[ItemRef]:
    """The newest item, then items in budget order while the running size stays within the budget."""
    walk: Final = chain.from_iterable(_newest_first_within_count(sizes, limits, kind) for kind in _SIZE_BUDGET_ORDER)
    others: Final = tuple(ref for ref in walk if ref != newest)
    totals: Final = accumulate(
        (sizes[ref.kind][ref.position] for ref in others), initial=sizes[newest.kind][newest.position]
    )
    fitting: Final = sum(1 for _ in takewhile(lambda total: total <= budget, islice(totals, 1, None)))
    return frozenset((newest, *others[:fitting]))


def plan_caps(sizes: Mapping[ItemKind, Sequence[int]], caps: OvalixCaps) -> CapPlan:
    """Apply the per-call caps to a call whose item sizes are listed per kind, oldest first."""
    every_item: Final = frozenset(chain.from_iterable(_refs_of_kind(sizes, kind) for kind in ItemKind))
    limits: Final = _count_limits(caps)
    over_count: Final = tuple((cap, kind) for cap, kind in _COUNT_CAPS if len(sizes.get(kind, ())) > limits[kind])
    over_size: Final = sum(map(sum, sizes.values())) > caps.call_size_bytes
    hits: Final = (*(cap for cap, _ in over_count), *((CapLimit.CALL_SIZE,) if over_size else ()))
    if not hits:
        return CapPlan(kept=every_item)
    newest: Final = _newest(sizes, over_count[0][1] if over_count else None)
    if caps.hit_behavior in ("block", "allow"):
        return CapPlan(frozenset({newest}), newest, hits[0], caps.hit_behavior)
    kept: Final = _kept_within_size(sizes, limits, newest, caps.call_size_bytes)
    if caps.hit_behavior == "trim" and any(ref.kind not in _TRIMMABLE_KINDS for ref in every_item - kept):
        return CapPlan(frozenset({newest}), newest, hits[0], "block")
    return CapPlan(kept, newest, hits[0], caps.hit_behavior)
