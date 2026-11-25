"""Pydantic config model for the Ovalix guardrail (Tracker API, application and checkpoint IDs)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import GuardrailConfigModel


class OvalixGuardrailConfigModel(GuardrailConfigModel):
    """Configuration parameters for the Ovalix guardrail (pre/post call checkpoints)."""

    tracker_api_base: str | None = Field(
        default=None,
        description="Base URL for the Ovalix Tracker service.",
    )
    tracker_api_key: str | None = Field(
        default=None,
        description="API key for the Ovalix Tracker service.",
    )
    application_id: str | None = Field(
        default=None,
        description="Application ID for the Ovalix Tracker service.",
    )
    pre_checkpoint_id: str | None = Field(
        default=None,
        description="Pre-checkpoint ID for the Ovalix Tracker service.",
    )
    post_checkpoint_id: str | None = Field(
        default=None,
        description="Post-checkpoint ID for the Ovalix Tracker service.",
    )
    file_checkpoint_id: str | None = Field(
        default=None,
        description="File-checkpoint ID for the Ovalix Tracker service (falls back to pre/post).",
    )
    pre_tool_checkpoint_id: str | None = Field(
        default=None,
        description="Pre-tool-use checkpoint ID for tool calls, in requests and responses (falls back to pre/post).",
    )
    post_tool_checkpoint_id: str | None = Field(
        default=None,
        description="Post-tool-use checkpoint ID for tool results (falls back to pre/post).",
    )
    streaming_transform_mode: Literal["block_only", "incremental_diff"] | None = Field(
        default=None,
        description=(
            "'block_only' (default) streams the response as the model sends it and only a block stops it. "
            "'incremental_diff' streams the text as Ovalix returns it, so anonymized text reaches the client, "
            "holding back each text's last 512 characters until the next check or the end of the stream."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        """Display name for this guardrail in the proxy UI."""
        return "Ovalix Guardrail"


class OvalixGuardrailStreamingParams(BaseModel):
    """Streaming settings read from the top level of an Ovalix guardrail's litellm_params.

    They stay undeclared on the config model: Bedrock, Noma, CrowdStrike and Akto read the same keys from
    LitellmParams' extras, which a declared field would empty for every guardrail.
    """

    model_config = ConfigDict(frozen=True)

    streaming_end_of_stream_only: bool | None = None
    streaming_sampling_rate: int | None = Field(default=None, ge=1)
    streaming_buffer_until_moderated: bool | None = None
