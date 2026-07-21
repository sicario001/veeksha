"""Base class for trace flavor generators."""

import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, cast

import pandas as pd

from veeksha.config.generator.session import (
    BaseTraceFlavorConfig,
    TraceSessionGeneratorConfig,
)
from veeksha.core.request import Request
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.requested_output import RequestedOutputSpec, TextOutputSpec
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.session_graph import (
    SessionEdge,
    SessionGraph,
    SessionNode,
    add_edge,
    add_node,
    topological_order,
)
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.types import ChannelModality


@dataclass(frozen=True)
class TraceNodeContext:
    node_id: int
    parent_nodes: List[int]
    history_parent: Optional[int]
    wait_after_ready: float


class TraceFlavorGeneratorBase:
    """Base class for trace flavor generators.

    Subclasses implement flavor-specific logic for:
    - Required trace columns validation
    - Prompt preparation from trace rows
    - Wrapping/epoch logic for looping
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        flavor_config: BaseTraceFlavorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        self.config = config
        self.flavor_config = flavor_config
        self.seed_manager = seed_manager
        self.tokenizer_provider = tokenizer_provider
        self.tokenizer = tokenizer_provider.for_modality(ChannelModality.TEXT)

        self._validate_trace_exists(config.trace_file)
        self.trace_df = self._load_trace(config.trace_file)
        self._validate_trace()

        # wrapping state
        self._num_wraps = 0
        self._session_groups: Optional[Iterator] = None
        self._current_session_id = 0
        self._current_request_id = 0

        self._rng = seed_manager.random("trace_shuffling")

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """Columns required in the trace DataFrame."""

    @abstractmethod
    def prepare_session(self, group: pd.DataFrame) -> Session:
        """Prepare a Session from a trace session group."""

    @abstractmethod
    def wrap(self) -> pd.DataFrame:
        """Wrap the trace for a new epoch."""

    def _wrap_shuffled(self) -> pd.DataFrame:
        """Shared epoch wrap: re-key session ids past the max, then shuffle."""
        df = self.trace_df.copy()
        max_sid = int(df["session_id"].max()) if not df.empty else 0
        df["session_id"] = df["session_id"] + max_sid + 1
        return self._shuffle_sessions(df)

    def get_warmup_sessions(self) -> List[Session]:
        """Return warmup sessions. Default empty, override for RAG."""
        return []

    def _load_trace(self, trace_file: str) -> pd.DataFrame:
        """Load trace from JSONL or CSV based on file extension."""
        ext = os.path.splitext(trace_file)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(trace_file)
            # Normalize common column name variations
            col_renames = {
                "num_prefill_tokens": "input_length",
                "num_decode_tokens": "output_length",
            }
            df = df.rename(
                columns={k: v for k, v in col_renames.items() if k in df.columns}
            )
            return df
        else:
            return pd.read_json(trace_file, lines=True)

    def _validate_trace_exists(self, trace_file: str):
        """Validate that trace file exists."""
        if not os.path.exists(trace_file):
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

    def _validate_trace(self):
        """Validate that required columns exist in trace."""
        for col in self.required_columns:
            if col not in self.trace_df.columns:
                raise ValueError(
                    f"Trace missing required column '{col}'. "
                    f"Required: {self.required_columns}"
                )

    def _get_session_groups(self) -> Iterator:
        """Get iterator over session groups."""
        return iter(self.trace_df.groupby("session_id", sort=False))

    def _next_session_id(self) -> int:
        """Get next global session ID."""
        sid = self._current_session_id
        self._current_session_id += 1
        return sid

    def _next_request_id(self) -> int:
        """Get next global request ID."""
        rid = self._current_request_id
        self._current_request_id += 1
        return rid

    def generate_session(self) -> Session:
        """Generate the next session from the trace."""
        if self._session_groups is None:
            self._session_groups = self._get_session_groups()

        try:
            _, group = next(self._session_groups)
            return self.prepare_session(group)
        except StopIteration:
            if self.config.wrap_mode:
                self.trace_df = self.wrap()
                self._num_wraps += 1
                self._session_groups = self._get_session_groups()
                _, group = next(self._session_groups)
                return self.prepare_session(group)
            else:
                raise StopIteration("Trace exhausted and wrap_mode is False")

    def capacity(self) -> int:
        """Return -1 (unbounded) if wrap mode, else session count."""
        if self.config.wrap_mode:
            return -1
        return len(self.trace_df.groupby("session_id"))

    def _build_linear_session_graph(
        self,
        num_requests: int,
        wait_times: List[float],
        is_history_parent: bool = True,
    ) -> SessionGraph:
        """Build a linear session graph (1→2→3...)."""
        graph = SessionGraph()
        for i in range(num_requests):
            wait = wait_times[i] if i < len(wait_times) else 0.0
            add_node(graph, SessionNode(id=i, wait_after_ready=wait))
            if i > 0:
                add_edge(
                    graph,
                    SessionEdge(src=i - 1, dst=i, is_history_parent=is_history_parent),
                )
        return graph

    def _build_session_graph_from_contexts(
        self, node_contexts: List[TraceNodeContext]
    ) -> SessionGraph:
        """Build a session graph from normalized trace node contexts."""
        graph = SessionGraph()
        seen_nodes = set()

        for ctx in node_contexts:
            if ctx.node_id in seen_nodes:
                raise ValueError(f"Duplicate node_id {ctx.node_id} in session trace.")
            seen_nodes.add(ctx.node_id)

            if len(set(ctx.parent_nodes)) != len(ctx.parent_nodes):
                raise ValueError(
                    f"node_id {ctx.node_id} contains duplicate parent_nodes."
                )
            if (
                ctx.history_parent is not None
                and ctx.history_parent not in ctx.parent_nodes
            ):
                raise ValueError(
                    f"node_id {ctx.node_id} has history_parent={ctx.history_parent} "
                    "which is not present in parent_nodes."
                )

            add_node(
                graph,
                SessionNode(id=ctx.node_id, wait_after_ready=ctx.wait_after_ready),
            )

        for ctx in node_contexts:
            for parent_node in ctx.parent_nodes:
                add_edge(
                    graph,
                    SessionEdge(
                        src=parent_node,
                        dst=ctx.node_id,
                        is_history_parent=parent_node == ctx.history_parent,
                    ),
                )

        topological_order(graph)
        return graph

    @staticmethod
    def _create_session_context(
        node_id: int,
        wait_after_ready: float,
        parent_nodes: Optional[List[int]] = None,
        history_parent: Optional[int] = None,
    ) -> Dict[str, object]:
        parent_nodes = list(parent_nodes or [])
        if history_parent is not None and history_parent not in parent_nodes:
            raise ValueError(
                f"history_parent={history_parent} is not in parent_nodes={parent_nodes}"
            )
        return {
            "node_id": node_id,
            "wait_after_ready": wait_after_ready,
            "parent_nodes": parent_nodes,
            "history_parent": history_parent,
        }

    def _create_text_request(
        self,
        node_id: int,
        prompt_text: str,
        target_output_tokens: int,
        wait_after_ready: float,
        parent_nodes: Optional[List[int]] = None,
        history_parent: Optional[int] = None,
        target_prompt_tokens: Optional[int] = None,
    ) -> Request:
        """Create a text-only Request and attach output spec."""

        channels = {
            ChannelModality.TEXT: TextChannelRequestContent(
                input_text=prompt_text,
                target_prompt_tokens=target_prompt_tokens,
            )
        }
        session_context = self._create_session_context(
            node_id=node_id,
            wait_after_ready=wait_after_ready,
            parent_nodes=parent_nodes,
            history_parent=history_parent,
        )
        requested_output = RequestedOutputSpec(
            text=TextOutputSpec(target_tokens=target_output_tokens)
        )
        return Request(
            id=self._next_request_id(),
            channels=channels,  # type: ignore
            session_context=session_context,
            requested_output=requested_output,
        )

    def _shuffle_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shuffle session order in DataFrame."""
        sid_order = df["session_id"].unique().tolist()
        self._rng.shuffle(sid_order)
        df_shuffled = pd.concat(
            [df[df["session_id"] == sid] for sid in sid_order]
        ).reset_index(drop=True)
        return cast(pd.DataFrame, df_shuffled)
