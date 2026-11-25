"""Ovalix guardrail integration: pre- and post-call checks via the Tracker service.

Use Ovalix Guardrails for your LLM calls. Supports pre_call (user input) and
post_call (model output) checkpoints with optional correction/blocking.
"""

import asyncio
import base64
import datetime
import gzip
import hashlib
import mimetypes
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, TypeVar

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.core_helpers import SupportsGet, as_str_mapping, require_get
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_caps import (
    CapHandling,
    CapLimit,
    CapPlan,
    ItemKind,
    ItemRef,
    OvalixCaps,
    plan_caps,
)
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix_extraction import (
    FilePart,
    extract_file_parts_from_images,
    extract_file_parts_from_messages,
    extract_tool_calls_from_messages,
    extract_tool_results,
    make_tool_data,
    str_or_none,
    tool_call_to_tool_data,
    tool_data_key,
    tool_result_text_indices,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import OvalixGuardrailConfigModel


BLOCKED_BY_OVALIX_FALLBACK_MESSAGE: Final = "This message was blocked by Ovalix"
BLOCKED_ACTION_TYPE: Final = "block"
_MODIFY_ACTION_TYPES: Final = ("anonymize", "sanitize")
_CAPS_TTL_SECONDS: Final = 300.0
_TRIM_PLACEHOLDER: Final = "[Removed: over the inspection limit]"
_DEFAULT_STREAMING_SAMPLING_RATE: Final = 5
_STREAM_HOLDBACK_CHARS: Final = 512
_NO_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_JSON: Final = TypeAdapter(object)
_FILE_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because file content anonymization isn't possible via LiteLLM"
)
_UNDECODABLE_FILE_BLOCK_REASON: Final = (
    "This message was blocked by Ovalix because an attached file couldn't be decoded for inspection"
)
_TOOL_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because tool call anonymization isn't possible via LiteLLM"
)
_TOOL_RESULT_BLOCK_ESCALATION_REASON: Final = (
    "This message was blocked by Ovalix because tool result anonymization isn't possible via LiteLLM"
)
_KEY_HASH_PREFIX_CHARS: Final = 12
_SHA256_HEX_DIGEST: Final = re.compile(r"[0-9a-f]{64}")


class _Identity(NamedTuple):
    identifier: str
    source: str


_NAMED_IDENTITY_FIELDS: Final = (
    ("user_api_key_user_email", "email"),
    ("user_api_key_user_id", "user-id"),
    ("user_api_key_end_user_id", "end-user-id"),
)
_UNKNOWN_IDENTITY: Final = _Identity("unknown", "unknown")


@dataclass(frozen=True, slots=True)
class _Caller:
    """Who a checkpoint request is on behalf of."""

    actor: str
    identity_source: str
    session_id: str


@dataclass(frozen=True, slots=True)
class _ItemTag:
    """What a checkpoint request carries about its item, and on the newest item of a capped call the cap it hit."""

    item_kind: ItemKind
    stream_id: str | None = None
    cap_hit: CapLimit | None = None
    cap_handling: CapHandling | None = None

    def wire_fields(self) -> Mapping[str, object]:
        stream_fields: Final = MappingProxyType({"stream_id": self.stream_id}) if self.stream_id else _NO_METADATA
        cap_fields: Final = (
            MappingProxyType({"cap_hit": self.cap_hit.value, "cap_handling": self.cap_handling})
            if self.cap_hit is not None
            else _NO_METADATA
        )
        return MappingProxyType({"item_kind": self.item_kind.value, **stream_fields, **cap_fields})


@dataclass(frozen=True, slots=True)
class _Routes:
    """The checkpoint each item kind goes to in one direction; None leaves that kind uninspected."""

    message: str | None
    tool_call: str | None
    tool_result: str | None
    file: str | None


@dataclass(frozen=True, slots=True)
class _Call:
    """The items of one call, each kind oldest first; a message is its position in `texts` and its text."""

    messages: tuple[tuple[int, str], ...]
    tool_calls: tuple[Mapping[str, object], ...]
    tool_results: tuple[Mapping[str, object], ...]
    files: tuple[FilePart, ...]

    def sizes(self) -> Mapping[ItemKind, Sequence[int]]:
        return MappingProxyType(
            {
                ItemKind.MESSAGE: tuple(_utf8_size(text) for _, text in self.messages),
                ItemKind.TOOL_CALL: tuple(map(_tool_content_size, self.tool_calls)),
                ItemKind.TOOL_RESULT: tuple(map(_tool_content_size, self.tool_results)),
                ItemKind.FILE: tuple(part.size for part in self.files),
            }
        )


@dataclass(frozen=True, slots=True)
class _Tagger:
    """Tags each item of a call: its kind, its stream id when it is part of a response, and the cap flags."""

    plan: CapPlan
    call: _Call
    stream_call_id: str | None

    def tag(self, ref: ItemRef) -> _ItemTag:
        flagged: Final = ref == self.plan.newest
        return _ItemTag(
            ref.kind,
            self._stream_id(ref),
            self.plan.cap_hit if flagged else None,
            self.plan.handling if flagged else None,
        )

    def _stream_id(self, ref: ItemRef) -> str | None:
        """`<litellm_call_id>:<kind>:<index>`, so every scan of one streamed item shares a key."""
        if self.stream_call_id is None:
            return None
        match ref.kind:
            case ItemKind.MESSAGE:
                return f"{self.stream_call_id}:text:{self.call.messages[ref.position][0]}"
            case ItemKind.TOOL_CALL:
                return f"{self.stream_call_id}:tool_call:{ref.position}"
            case ItemKind.TOOL_RESULT | ItemKind.FILE:
                return None


_T = TypeVar("_T")


def _encode_file_wire_format(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw)).decode()


def _utf8_size(text: str) -> int:
    return len(text.encode())


def _tool_content_size(data: Mapping[str, object]) -> int:
    return _utf8_size(str_or_none(data.get("content")) or "")


def _may_hold_items(inputs: GenericGuardrailAPIInputs) -> bool:
    return bool(
        inputs.get("texts") or inputs.get("tool_calls") or inputs.get("structured_messages") or inputs.get("images")
    )


def _unique_tool_calls(tool_calls: Sequence[object]) -> tuple[Mapping[str, object], ...]:
    unique: Final = MappingProxyType(
        {tool_data_key(data): data for data in (tool_call_to_tool_data(tc) for tc in tool_calls) if data}
    )
    return tuple(unique.values())


def _file_parts(inputs: GenericGuardrailAPIInputs, is_response: bool, size_limit: int) -> tuple[FilePart, ...]:
    if is_response:
        return extract_file_parts_from_images(inputs.get("images"), size_limit=size_limit)
    return extract_file_parts_from_messages(inputs.get("structured_messages") or (), size_limit=size_limit)


def _messages(texts: Sequence[str], tool_result_texts: frozenset[int]) -> tuple[tuple[int, str], ...]:
    return tuple((index, text) for index, text in enumerate(texts) if text.strip() and index not in tool_result_texts)


def _extract_call(inputs: GenericGuardrailAPIInputs, is_response: bool, routes: _Routes, file_size_limit: int) -> _Call:
    """The call's items of every kind that has a checkpoint in this direction."""
    structured_messages: Final = inputs.get("structured_messages") or ()
    texts: Final = inputs.get("texts") or ()
    tool_calls: Final = (*(inputs.get("tool_calls") or ()), *extract_tool_calls_from_messages(structured_messages))
    tool_results: Final = extract_tool_results(structured_messages) if routes.tool_result else ()
    return _Call(
        messages=_messages(texts, tool_result_text_indices(structured_messages, texts)) if routes.message else (),
        tool_calls=_unique_tool_calls(tool_calls) if routes.tool_call else (),
        tool_results=tuple(make_tool_data(name, content) for name, content, _ in tool_results),
        files=_file_parts(inputs, is_response, file_size_limit) if routes.file else (),
    )


def _kept(tagger: _Tagger, kind: ItemKind, items: Sequence[_T]) -> tuple[tuple[_T, _ItemTag], ...]:
    """The items of one kind the plan inspects, oldest first, each with its request tag."""
    refs: Final = tuple(ItemRef(kind, position) for position in range(len(items)))
    return tuple((items[ref.position], tagger.tag(ref)) for ref in refs if ref in tagger.plan.kept)


def _trimmed_text_indices(plan: CapPlan, messages: Sequence[tuple[int, str]]) -> frozenset[int]:
    """Positions in `texts` of the messages a trim dropped, which reach the model as the placeholder."""
    if plan.handling != "trim":
        return frozenset()
    refs: Final = tuple(ItemRef(ItemKind.MESSAGE, position) for position in range(len(messages)))
    return frozenset(messages[ref.position][0] for ref in refs if ref not in plan.kept)


def _with_stream_holdback(result: GenericGuardrailAPIInputs) -> GenericGuardrailAPIInputs:
    """Withhold the end of each response text from streaming, so the next check can still rewrite it."""
    texts: Final = result.get("texts") or ()
    if not texts:
        return result
    return {**result, "stream_holdback_chars": [min(len(text), _STREAM_HOLDBACK_CHARS) for text in texts]}


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    return as_str_mapping(value) or _NO_METADATA


def _stripped(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _named_identities(metadata: Mapping[str, object]) -> Iterator[_Identity]:
    for field, source in _NAMED_IDENTITY_FIELDS:
        value = _stripped(metadata.get(field))
        if value:
            yield _Identity(value, source)


def _key_hash_prefix(key_hash: str) -> str:
    """The first hex chars of the key's SHA-256; LiteLLM leaves some keys (JWTs, non-`sk-` keys) unhashed here."""
    digest: Final = (
        key_hash if _SHA256_HEX_DIGEST.fullmatch(key_hash) else hashlib.sha256(key_hash.encode()).hexdigest()
    )
    return digest[:_KEY_HASH_PREFIX_CHARS]


def _resolve_identity(data: Mapping[str, object]) -> _Identity:
    """The request's actor: the key's user email, user id, end-user id or key hash, whichever is set first."""
    metadata: Final = _mapping_or_empty(data.get("metadata") or data.get("litellm_metadata"))
    named: Final = next(_named_identities(metadata), None)
    if named is not None:
        return named
    key_hash: Final = _stripped(metadata.get("user_api_key_hash"))
    if key_hash:
        return _Identity(f"api-key:{_key_hash_prefix(key_hash)}", "api-key")
    return _UNKNOWN_IDENTITY


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    """Base-class constructor options this guardrail forwards untouched to CustomGuardrail."""


class OvalixGuardrailMissingSecrets(Exception):
    """Raised when required Ovalix config (API base, key, application/checkpoint IDs) is missing."""


class OvalixGuardrailBlockedException(GuardrailRaisedException):
    """
    Raised when Ovalix blocks a message. Sets status_code=400 so the proxy
    returns 400 and HTTP clients do not retry (they retry on 5xx).
    """

    status_code = 400

    def __init__(
        self,
        guardrail_name: str | None = None,
        message: str = "",
        should_wrap_with_default_message: bool = True,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            message=message,
            should_wrap_with_default_message=should_wrap_with_default_message,
            blocked_content=True,
        )


class OvalixGuardrail(CustomGuardrail):
    """
    Ovalix guardrail: pre-prompt (pre_call) and post-prompt (post_call) checks
    via the Tracker service, with application and checkpoint resolution from the
    Monolith backend.
    """

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        tracker_api_base: str | None = None,
        tracker_api_key: str | None = None,
        application_id: str | None = None,
        pre_checkpoint_id: str | None = None,
        post_checkpoint_id: str | None = None,
        file_checkpoint_id: str | None = None,
        pre_tool_checkpoint_id: str | None = None,
        post_tool_checkpoint_id: str | None = None,
        streaming_end_of_stream_only: bool | None = None,
        streaming_sampling_rate: int | None = None,
        streaming_transform_mode: Literal["block_only", "incremental_diff"] | None = None,
        streaming_buffer_until_moderated: bool | None = None,
        supported_event_hooks: list[GuardrailEventHooks] | None = None,
        async_handler: AsyncHTTPHandler | None = None,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Unpack[_CustomGuardrailOptions],
    ):
        self._tracker_api_base = tracker_api_base or os.environ.get("OVALIX_TRACKER_API_BASE")
        self._tracker_api_key = tracker_api_key or os.environ.get("OVALIX_TRACKER_API_KEY")
        self._application_id = application_id or os.environ.get("OVALIX_APPLICATION_ID")
        self._pre_checkpoint_id = pre_checkpoint_id or os.environ.get("OVALIX_PRE_CHECKPOINT_ID")
        self._post_checkpoint_id = post_checkpoint_id or os.environ.get("OVALIX_POST_CHECKPOINT_ID")
        self._file_checkpoint_id = file_checkpoint_id or os.environ.get("OVALIX_FILE_CHECKPOINT_ID")
        self._pre_tool_checkpoint_id = pre_tool_checkpoint_id or os.environ.get("OVALIX_PRE_TOOL_CHECKPOINT_ID")
        self._post_tool_checkpoint_id = post_tool_checkpoint_id or os.environ.get("OVALIX_POST_TOOL_CHECKPOINT_ID")
        self.streaming_end_of_stream_only: bool = bool(streaming_end_of_stream_only)
        self.streaming_sampling_rate: int = (
            _DEFAULT_STREAMING_SAMPLING_RATE if streaming_sampling_rate is None else streaming_sampling_rate
        )
        self.streaming_transform_mode: Literal["block_only", "incremental_diff"] = (
            "block_only" if streaming_transform_mode is None else streaming_transform_mode
        )
        if streaming_buffer_until_moderated is not None:
            self.streaming_buffer_until_moderated: bool = streaming_buffer_until_moderated

        event_hooks: Final = self._validated_event_hooks(supported_event_hooks or ())

        self._tracker_headers = dict(
            httpx.Headers(
                MappingProxyType(
                    {
                        "Authorization": f"Bearer {self._tracker_api_key}",
                        "x-api-key": self._tracker_api_key or "",
                        "Content-Type": "application/json",
                    }
                ),
                encoding="utf-8",
            )
        )

        self._async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self._clock = clock
        self._caps: OvalixCaps | None = None
        self._caps_checked_at = 0.0

        super().__init__(supported_event_hooks=event_hooks, **kwargs)
        verbose_proxy_logger.debug(
            "Ovalix Guardrail initialized: tracker=%s, application_id=%s, pre_checkpoint_id=%s, post_checkpoint_id=%s, "
            "pre_tool_checkpoint_id=%s, post_tool_checkpoint_id=%s",
            self._tracker_api_base,
            self._application_id,
            self._pre_checkpoint_id,
            self._post_checkpoint_id,
            self._pre_tool_checkpoint_id,
            self._post_tool_checkpoint_id,
        )

    def _validated_event_hooks(self, requested: Sequence[GuardrailEventHooks]) -> list[GuardrailEventHooks]:
        """Ensure required secrets and checkpoint IDs are set; return the requested hooks plus one per configured checkpoint."""
        errors: Final = tuple(
            message
            for missing, message in (
                (not self._tracker_api_base, "Tracker API base, set OVALIX_TRACKER_API_BASE or pass tracker_api_base"),
                (not self._tracker_api_key, "Tracker API key, set OVALIX_TRACKER_API_KEY or pass tracker_api_key"),
                (not self._application_id, "Application ID, set OVALIX_APPLICATION_ID or pass application_id"),
                (
                    not self._pre_checkpoint_id and GuardrailEventHooks.pre_call in requested,
                    "Pre-checkpoint ID, set OVALIX_PRE_CHECKPOINT_ID or pass pre_checkpoint_id",
                ),
                (
                    not self._post_checkpoint_id and GuardrailEventHooks.post_call in requested,
                    "Post-checkpoint ID, set OVALIX_POST_CHECKPOINT_ID or pass post_checkpoint_id",
                ),
                (
                    not self._pre_checkpoint_id and not self._post_checkpoint_id,
                    "Pre-checkpoint ID or Post-checkpoint ID, set OVALIX_PRE_CHECKPOINT_ID or OVALIX_POST_CHECKPOINT_ID or pass pre_checkpoint_id or post_checkpoint_id",
                ),
            )
            if missing
        )

        if errors:
            raise OvalixGuardrailMissingSecrets("Missing Ovalix guardrail configuration errors: " + ". ".join(errors))

        auto_added: Final = tuple(
            hook
            for configured, hook in (
                (self._pre_checkpoint_id, GuardrailEventHooks.pre_call),
                (self._post_checkpoint_id, GuardrailEventHooks.post_call),
            )
            if configured and hook not in requested
        )
        return [*requested, *auto_added]

    def _get_tracker_actor_id(self, data: Mapping[str, object]) -> str:
        """Normalize the actor string into a short, stable id for Tracker API payloads."""
        # NOTE: this hash is purely for normalization — it collapses an arbitrary actor
        # string (email, user id, end-user id, key hash prefix, or "unknown") into a compact, fixed-length, consistent
        # key. It is not a privacy/security measure and the actor value is not sensitive,
        # so a plain SHA-256 (truncated) is sufficient; no salting/KDF is needed here.
        actor_id: Final = _resolve_identity(data).identifier.encode()
        normalized_actor_id: Final = hashlib.sha256(actor_id).hexdigest()[:8]
        return normalized_actor_id

    def _get_session_id(self, data: Mapping[str, object]) -> str:
        """Return a unique identifier for the chat/session (actor + date + application_id)."""
        actor_hash: Final = self._get_tracker_actor_id(data)
        today: Final = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        return f"{actor_hash}_{today}_{self._application_id}"

    def _routes(self, is_response: bool) -> _Routes:
        prompt_checkpoint: Final = self._post_checkpoint_id if is_response else self._pre_checkpoint_id
        return _Routes(
            message=prompt_checkpoint,
            tool_call=self._pre_tool_checkpoint_id or prompt_checkpoint,
            tool_result=self._post_tool_checkpoint_id or prompt_checkpoint,
            file=self._file_checkpoint_id or prompt_checkpoint,
        )

    async def _fetch_caps(self) -> OvalixCaps:
        response: Final = await self._async_handler.get(
            f"{self._tracker_api_base}/tracking/beta/caps", headers=self._tracker_headers, timeout=self.timeout
        )
        response.raise_for_status()
        return OvalixCaps.model_validate(response.json())

    async def _current_caps(self) -> OvalixCaps:
        """The org's caps, refetched once the copy in hand is older than the TTL and kept when a refetch fails."""
        now: Final = self._clock()
        cached: Final = self._caps
        if cached is not None and now - self._caps_checked_at < _CAPS_TTL_SECONDS:
            return cached
        try:
            fetched: Final = await self._fetch_caps()
        except Exception as e:
            if cached is None:
                verbose_proxy_logger.exception("Ovalix caps fetch failed: %s", e)
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=f"Ovalix guardrail error: {e}",
                    should_wrap_with_default_message=False,
                ) from e
            verbose_proxy_logger.warning("Ovalix caps fetch failed, keeping the last caps: %s", e)
            self._caps_checked_at = now
            return cached
        self._caps = fetched
        self._caps_checked_at = now
        return fetched

    async def _call_checkpoint(
        self,
        data_type: str,
        data: Mapping[str, object],
        checkpoint_id: str | None,
        caller: _Caller,
        tag: _ItemTag,
        oversize: bool = False,
    ) -> object:
        """Call the Ovalix Tracker checkpoint API and return the JSON response.

        Both routes live on the tracker's /beta litellm router, which accepts the api key this
        guardrail already sends.
        """
        application_id: Final = self._application_id
        if not application_id or not checkpoint_id:
            raise ValueError("Ovalix: application_id or checkpoint_id not resolved")

        route: Final = "file_checkpoint" if data_type == "FILE" else "checkpoint"
        payload: Final = {
            "application_id": application_id,
            "checkpoint_id": checkpoint_id,
            "actor": caller.actor,
            "identity_source": caller.identity_source,
            "session_id": caller.session_id,
            "data_type": data_type,
            "data": data,
            "tool": "LiteLLM",
            **tag.wire_fields(),
            **(MappingProxyType({"oversize": True}) if oversize else _NO_METADATA),
        }
        response: Final = await self._async_handler.post(
            f"{self._tracker_api_base}/tracking/beta/{route}",
            headers=self._tracker_headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _DECODED_JSON.validate_python(response.json())

    def _verdict(self, resp: object) -> tuple[str, str | None]:
        decoded: Final = require_get(resp)
        return (str_or_none(decoded.get("action_type")) or "").lower(), self._get_trackers_corrected_message(decoded)

    async def _send(
        self,
        data_type: str,
        data: Mapping[str, object],
        checkpoint_id: str | None,
        caller: _Caller,
        tag: _ItemTag,
        oversize: bool = False,
    ) -> tuple[str, str | None]:
        """Send one item and return the tracker's action and corrected content; a failed call fails closed."""
        try:
            resp: Final = await self._call_checkpoint(data_type, data, checkpoint_id, caller, tag, oversize=oversize)
        except Exception as e:
            verbose_proxy_logger.exception("Ovalix checkpoint call failed: %s", e)
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=f"Ovalix guardrail error: {e}",
                should_wrap_with_default_message=False,
            ) from e
        return self._verdict(resp)

    async def _block_reason_for_item(
        self,
        data_type: str,
        data: Mapping[str, object],
        checkpoint_id: str | None,
        caller: _Caller,
        tag: _ItemTag,
        escalation_reason: str,
        oversize: bool = False,
    ) -> str | None:
        action, corrected = await self._send(data_type, data, checkpoint_id, caller, tag, oversize=oversize)
        if action == BLOCKED_ACTION_TYPE:
            return corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
        if action in _MODIFY_ACTION_TYPES:
            return escalation_reason
        return None

    async def _check_tool_items(
        self,
        items: Sequence[tuple[Mapping[str, object], _ItemTag]],
        checkpoint_id: str | None,
        caller: _Caller,
        escalation_reason: str,
    ) -> str | None:
        for data, tag in items:
            reason = await self._block_reason_for_item("TOOL", data, checkpoint_id, caller, tag, escalation_reason)
            if reason is not None:
                return reason
        return None

    async def _check_files_for_block(
        self,
        file_parts: Sequence[tuple[FilePart, _ItemTag]],
        checkpoint_id: str | None,
        caller: _Caller,
    ) -> str | None:
        for part, tag in sorted(file_parts, key=lambda kept: kept[0].message_index, reverse=True):
            data = await self._file_part_to_data(part)
            reason = await self._block_reason_for_item(
                "FILE",
                data,
                checkpoint_id,
                caller,
                tag,
                _FILE_BLOCK_ESCALATION_REASON,
                oversize=part.oversize,
            )
            if reason is not None:
                return reason
        return None

    async def _send_item(
        self, call: _Call, ref: ItemRef, routes: _Routes, caller: _Caller, tag: _ItemTag
    ) -> tuple[str, str | None]:
        match ref.kind:
            case ItemKind.MESSAGE:
                return await self._send(
                    "TEXT", {"content": call.messages[ref.position][1]}, routes.message, caller, tag
                )
            case ItemKind.TOOL_CALL:
                return await self._send("TOOL", call.tool_calls[ref.position], routes.tool_call, caller, tag)
            case ItemKind.TOOL_RESULT:
                return await self._send("TOOL", call.tool_results[ref.position], routes.tool_result, caller, tag)
            case ItemKind.FILE:
                part: Final = call.files[ref.position]
                data: Final = await self._file_part_to_data(part)
                return await self._send("FILE", data, routes.file, caller, tag, oversize=part.oversize)

    async def _check_newest_only(self, tagger: _Tagger, newest: ItemRef, routes: _Routes, caller: _Caller) -> None:
        """Send only the newest item, flagged with the cap the call hit; the tracker's verdict on it decides the call."""
        action, corrected = await self._send_item(tagger.call, newest, routes, caller, tagger.tag(newest))
        if action == BLOCKED_ACTION_TYPE:
            self._block_current_message(corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE)

    async def _check_block_only_items(self, tagger: _Tagger, routes: _Routes, caller: _Caller) -> None:
        call: Final = tagger.call
        file_block: Final = await self._check_files_for_block(
            _kept(tagger, ItemKind.FILE, call.files), routes.file, caller
        )
        if file_block is not None:
            self._block_current_message(file_block)
        tool_block: Final = await self._check_tool_items(
            _kept(tagger, ItemKind.TOOL_CALL, call.tool_calls), routes.tool_call, caller, _TOOL_BLOCK_ESCALATION_REASON
        )
        if tool_block is not None:
            self._block_current_message(tool_block)
        tool_result_block: Final = await self._check_tool_items(
            _kept(tagger, ItemKind.TOOL_RESULT, call.tool_results),
            routes.tool_result,
            caller,
            _TOOL_RESULT_BLOCK_ESCALATION_REASON,
        )
        if tool_result_block is not None:
            self._block_current_message(tool_result_block)

    def scans_each_text_separately(self) -> bool:
        return False

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        identity: Final = _resolve_identity(request_data)
        caller: Final = _Caller(identity.identifier, identity.source, self._get_session_id(request_data))
        is_response: Final = input_type == "response"
        routes: Final = self._routes(is_response)
        if not (routes.file or routes.tool_call or routes.tool_result):
            verbose_proxy_logger.debug(
                "Ovalix guardrail: application %s has no %s checkpoint, leaving this direction uninspected",
                self._application_id,
                input_type,
            )
            return inputs
        if not _may_hold_items(inputs):
            return inputs

        caps: Final = await self._current_caps()
        call: Final = _extract_call(inputs, is_response, routes, caps.file_size_bytes)
        if any(part.undecodable for part in call.files):
            self._block_current_message(_UNDECODABLE_FILE_BLOCK_REASON)
        plan: Final = plan_caps(call.sizes(), caps)
        tagger: Final = _Tagger(plan, call, str_or_none(request_data.get("litellm_call_id")) if is_response else None)
        if plan.newest is not None and plan.newest_only:
            await self._check_newest_only(tagger, plan.newest, routes, caller)
            return _with_stream_holdback(inputs) if is_response else inputs
        await self._check_block_only_items(tagger, routes, caller)

        texts: Final = inputs.get("texts") or ()
        if not routes.message:
            if texts:
                verbose_proxy_logger.debug(
                    "Ovalix guardrail: application %s has no %s prompt checkpoint, skipping text inspection",
                    self._application_id,
                    input_type,
                )
            return inputs
        output_texts: Final = await self._check_texts(
            texts,
            _kept(tagger, ItemKind.MESSAGE, call.messages),
            _trimmed_text_indices(plan, call.messages),
            routes.message,
            caller,
            caps,
        )
        result: Final[GenericGuardrailAPIInputs] = inputs if output_texts is None else {**inputs, "texts": output_texts}
        return _with_stream_holdback(result) if is_response else result

    async def _file_part_to_data(self, part: FilePart) -> Mapping[str, object]:
        extension: Final = mimetypes.guess_extension(part.mime_hint) if part.mime_hint else None
        name: Final = part.name or (f"file{extension}" if extension else "file")
        content: Final = (
            await asyncio.get_event_loop().run_in_executor(None, _encode_file_wire_format, part.data)
            if part.data
            else None
        )
        return {"name": name, "content": content}

    async def _check_texts(
        self,
        texts: Sequence[str],
        messages: Sequence[tuple[tuple[int, str], _ItemTag]],
        trimmed: frozenset[int],
        checkpoint_id: str,
        caller: _Caller,
        caps: OvalixCaps,
    ) -> list[str] | None:
        """Check the kept messages newest first; trimmed positions reach the model as the placeholder.

        A blocked message blocks the call when it is the newest text, or when it is over the chars cap under a block
        handling. Any other blocked message is replaced by the block message.
        """
        over_cap_blocks: Final = caps.hit_behavior == "block"
        checked: Final = MappingProxyType(
            {
                index: await self._check_text(
                    text,
                    checkpoint_id,
                    caller,
                    tag,
                    blocks_call=index == len(texts) - 1 or (over_cap_blocks and len(text) > caps.message_chars),
                )
                for (index, text), tag in reversed(messages)
            }
        )
        output: Final = [
            _TRIM_PLACEHOLDER if index in trimmed else checked.get(index, text) for index, text in enumerate(texts)
        ]
        return output if output != list(texts) else None

    async def _check_text(
        self,
        content: str,
        checkpoint_id: str,
        caller: _Caller,
        tag: _ItemTag,
        blocks_call: bool,
    ) -> str:
        """Return the text that replaces `content` after Ovalix checks it; a block raises when it blocks the call."""
        action, corrected = await self._send("TEXT", {"content": content}, checkpoint_id, caller, tag)
        if action == BLOCKED_ACTION_TYPE:
            block_message: Final = corrected or BLOCKED_BY_OVALIX_FALLBACK_MESSAGE
            if blocks_call:
                self._block_current_message(block_message)
            return block_message
        if action in _MODIFY_ACTION_TYPES and corrected is not None:
            return corrected
        return content

    def _block_current_message(self, blocking_message: str) -> None:
        """Raise OvalixGuardrailBlockedException with the given message (no default wrapper)."""
        raise OvalixGuardrailBlockedException(
            guardrail_name=self.guardrail_name,
            message=blocking_message,
            should_wrap_with_default_message=False,
        )

    def _get_trackers_corrected_message(self, resp: SupportsGet) -> str | None:
        """Extract corrected/blocking message content from Tracker checkpoint response."""
        modified: Final = as_str_mapping(resp.get("modified_data"))
        if modified is None:
            return None
        content: Final = modified.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def get_config_model() -> type["OvalixGuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        return OvalixGuardrailConfigModel
