"""NeuroMem dashboard (Phase 7).

Six tabs backed by real memory_api endpoints:
- Overview: cluster + memory stats
- Memory Explorer: search + inspect
- Graph: relationship walk around a chosen memory (pyvis)
- Timeline: recent memories chronologically
- Retrieval Trace: recall with per-signal component scores
- Consolidation Trace: run a synchronous consolidation job and see the outcome
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from pyvis.network import Network

API_URL = os.getenv("MEMORY_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 30.0

st.set_page_config(page_title="NeuroMem", layout="wide")

st.title("NeuroMem POC")
st.caption(f"memory_api: `{API_URL}`")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(
    path: str, params: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT
) -> tuple[bool, Any]:
    try:
        r = httpx.get(f"{API_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except httpx.HTTPError as err:
        return False, {"error": str(err)}


def _post(path: str, json: dict[str, Any], timeout: float = DEFAULT_TIMEOUT) -> tuple[bool, Any]:
    try:
        r = httpx.post(f"{API_URL}{path}", json=json, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except httpx.HTTPError as err:
        return False, {
            "error": str(err),
            "status_code": getattr(err.response, "status_code", None)
            if hasattr(err, "response")
            else None,
        }


@st.cache_data(ttl=15, show_spinner=False)
def _cached_stats() -> dict[str, Any]:
    ok, data = _get("/memory/stats")
    return data if ok else {}


@st.cache_data(ttl=15, show_spinner=False)
def _cached_timeline(limit: int, memory_types: tuple[str, ...]) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if memory_types:
        params["memory_types"] = list(memory_types)
    ok, data = _get("/memory/timeline", params=params)
    return data if ok else {"memories": []}


def _status_of(payload: Any) -> str:
    if isinstance(payload, dict):
        s = payload.get("status")
        if isinstance(s, str):
            return s
    return "err"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


(
    tab_overview,
    tab_explorer,
    tab_graph,
    tab_timeline,
    tab_retrieval,
    tab_consolidation,
) = st.tabs(
    [
        "Overview",
        "Memory Explorer",
        "Graph",
        "Timeline",
        "Retrieval Trace",
        "Consolidation Trace",
    ]
)


# --------- Overview ---------
with tab_overview:
    st.subheader("Service health")
    cols = st.columns(4)

    ok_api, data_api = _get("/health")
    with cols[0]:
        st.metric("memory_api", "up" if ok_api else "down")
        st.json(data_api)

    ok_os, data_os = _get("/health/opensearch")
    with cols[1]:
        st.metric("OpenSearch", _status_of(data_os) if ok_os else "down")
        st.json(data_os)

    ok_llm, data_llm = _get("/health/llm")
    with cols[2]:
        st.metric("LLM (OpenAI)", _status_of(data_llm) if ok_llm else "down")
        st.json(data_llm)

    ok_emb, data_emb = _get("/health/embeddings")
    with cols[3]:
        st.metric("Embeddings", _status_of(data_emb) if ok_emb else "down")
        st.json(data_emb)

    st.divider()
    st.subheader("Memory stats")
    if st.button("Refresh stats", key="btn-refresh-stats"):
        _cached_stats.clear()

    stats = _cached_stats()
    if not stats:
        st.info("Stats unavailable — is `memory_api` reachable?")
    else:
        counts = stats.get("counts", {})
        totals = stats.get("totals", {})
        avgs = stats.get("averages", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total memories", totals.get("memories", 0))
        c2.metric("Episodic", counts.get("episodic", 0))
        c3.metric("Semantic", counts.get("semantic", 0))
        c4.metric("Procedural", counts.get("procedural", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Relationships", totals.get("relationships", 0))
        c2.metric("Avg confidence", f"{avgs.get('confidence', 0.0):.3f}")
        c3.metric("Avg activation", f"{avgs.get('activation', 0.0):.3f}")
        c4.metric("Avg importance", f"{avgs.get('importance', 0.0):.3f}")

        st.caption(
            "Averages are computed by OpenSearch across `neuromem-{episodic,semantic,procedural}` "
            "on every refresh."
        )


# --------- Memory Explorer ---------
with tab_explorer:
    st.subheader("Search memories")
    c1, c2 = st.columns([3, 1])
    query = c1.text_input(
        "Free-text search",
        key="explorer-query",
        placeholder="e.g. opensearch, jetson, deployment...",
    )
    limit = c2.number_input("Limit", min_value=1, max_value=100, value=25, key="explorer-limit")

    type_filter = st.multiselect(
        "Filter memory types",
        options=["episodic", "semantic", "procedural"],
        default=[],
        key="explorer-types",
    )

    if query.strip():
        params: dict[str, Any] = {"q": query, "limit": int(limit)}
        if type_filter:
            params["memory_types"] = type_filter
        ok, data = _get("/memory/search", params=params)
        if not ok:
            st.error(f"Search failed: {data.get('error')}")
        else:
            results = data.get("results", [])
            st.caption(f"{len(results)} results")
            if results:
                st.dataframe(
                    pd.DataFrame(results),
                    use_container_width=True,
                    hide_index=True,
                )
                st.divider()
                st.subheader("Inspect a memory")
                pick = st.selectbox(
                    "Memory ID",
                    options=[r["id"] for r in results],
                    key="explorer-pick",
                )
                if pick:
                    ok_mem, mem = _get(f"/memory/{pick}")
                    if ok_mem:
                        st.json(mem)
                    ok_hist, hist = _get(f"/memory/{pick}/history")
                    if ok_hist:
                        st.subheader("Lifecycle")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Activation", f"{hist.get('activation', 0.0):.3f}")
                        c2.metric(
                            "Effective act.",
                            f"{hist.get('effective_activation', 0.0):.3f}",
                        )
                        c3.metric("Access count", hist.get("access_count", 0))
                        c4.metric(
                            "Relationships",
                            len(hist.get("relationships", [])),
                        )
                        st.json(hist)
                    if st.button("Set as graph center", key="btn-set-graph-center"):
                        st.session_state["graph_center_id"] = pick
                        st.success(f"Graph center set to `{pick}` — open the Graph tab.")
    else:
        st.info("Enter a search term to find memories.")


# --------- Graph ---------
with tab_graph:
    st.subheader("Memory relationship graph")
    default_center = st.session_state.get("graph_center_id", "")
    c1, c2 = st.columns([3, 1])
    center = c1.text_input(
        "Center memory ID",
        value=default_center,
        key="graph-center",
        placeholder="epi_... / sem_... / pro_...",
    )
    depth = c2.slider("Depth", min_value=1, max_value=3, value=1, key="graph-depth")

    if center.strip():
        ok, data = _get("/memory/graph", params={"center_id": center.strip(), "depth": int(depth)})
        if not ok:
            st.error(f"Graph fetch failed: {data.get('error')}")
        else:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            st.caption(f"{len(nodes)} nodes, {len(edges)} edges")

            net = Network(
                height="520px", width="100%", directed=True, bgcolor="#111827", font_color="#f9fafb"
            )
            net.barnes_hut()
            color_by_type = {
                "episodic": "#f97316",
                "semantic": "#38bdf8",
                "procedural": "#a3e635",
            }
            for n in nodes:
                nid = n["id"]
                label = f"{nid}\n({n['memory_type']})"
                title = n["content"][:220]
                color = color_by_type.get(n["memory_type"], "#c084fc")
                border_width = 4 if nid == data.get("center_id") else 1
                net.add_node(nid, label=label, title=title, color=color, borderWidth=border_width)
            for e in edges:
                net.add_edge(
                    e["from_id"],
                    e["to_id"],
                    label=e["relationship_type"],
                    title=f"confidence={e['confidence']:.2f}",
                )
            html = net.generate_html(notebook=False)
            st.components.v1.html(html, height=550, scrolling=False)

            if edges:
                st.dataframe(pd.DataFrame(edges), use_container_width=True, hide_index=True)
    else:
        st.info("Enter a memory ID above, or set one from the Memory Explorer tab.")


# --------- Timeline ---------
with tab_timeline:
    st.subheader("Memory timeline")
    c1, c2 = st.columns([1, 3])
    tl_limit = c1.slider("Limit", min_value=10, max_value=200, value=50, key="timeline-limit")
    tl_types = c2.multiselect(
        "Memory types",
        options=["episodic", "semantic", "procedural"],
        default=[],
        key="timeline-types",
    )

    if st.button("Refresh timeline", key="btn-refresh-timeline"):
        _cached_timeline.clear()

    data = _cached_timeline(int(tl_limit), tuple(tl_types))
    memories = data.get("memories", [])
    if not memories:
        st.info("No memories to show.")
    else:
        df = pd.DataFrame(memories)
        st.dataframe(df, use_container_width=True, hide_index=True)


# --------- Retrieval Trace ---------
with tab_retrieval:
    st.subheader("Retrieval trace")
    query = st.text_area(
        "Query",
        key="retrieval-query",
        placeholder="Ask the memory engine anything...",
        height=100,
    )
    c1, c2, c3 = st.columns(3)
    top_k = c1.slider("Top-K", min_value=1, max_value=20, value=5, key="retrieval-k")
    include_rel = c2.checkbox("Include relationship boost", value=True, key="retrieval-rel")
    types = c3.multiselect(
        "Filter memory types",
        options=["episodic", "semantic", "procedural"],
        default=[],
        key="retrieval-types",
    )

    if st.button("Recall", key="btn-recall") and query.strip():
        body: dict[str, Any] = {
            "query": query.strip(),
            "limit": int(top_k),
            "include_relationships": include_rel,
        }
        if types:
            body["memory_types"] = types
        t0 = time.perf_counter()
        ok, data = _post("/memory/recall", json=body)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        st.caption(f"Wall-clock: {elapsed_ms:.0f} ms")
        if not ok:
            st.error(f"Recall failed: {data.get('error')}")
        else:
            weights = data.get("weights", {})
            with st.expander("Weights used", expanded=False):
                st.json(weights)
            results = data.get("memories", [])
            st.caption(f"{len(results)} results")
            for i, r in enumerate(results, start=1):
                with st.container(border=True):
                    top = st.columns([1, 4, 1])
                    top[0].markdown(f"**#{i}**")
                    top[1].markdown(f"**{r['memory_type']}** · `{r['id']}`")
                    top[2].markdown(f"score **{r['score']:.3f}**")
                    st.markdown(r["content"])
                    exp = r.get("explanation", {})
                    exp_df = pd.DataFrame(
                        [
                            {"signal": k, "component": v, "weight": weights.get(k, 0.0)}
                            for k, v in exp.items()
                        ]
                    )
                    exp_df["contribution"] = exp_df["component"] * exp_df["weight"]
                    st.bar_chart(exp_df.set_index("signal")["contribution"])
    elif query.strip() == "":
        st.info("Enter a query and press Recall.")


# --------- Consolidation Trace ---------
with tab_consolidation:
    st.subheader("Consolidation trace")
    st.caption(
        "Compose a synthetic consolidation job and run it end-to-end. "
        "The pipeline creates a turn `EpisodicMemory`, then processes each candidate "
        "with LLM-driven conflict adjudication."
    )
    user_msg = st.text_area(
        "User message",
        key="cons-user",
        height=80,
        placeholder="I switched from OpenSearch to Qdrant.",
    )
    assistant_msg = st.text_area(
        "Assistant reply",
        key="cons-assistant",
        height=80,
        placeholder="Noted.",
    )
    conv_id = st.text_input("Conversation ID", value="dashboard", key="cons-conv")

    if "cons_candidates" not in st.session_state:
        st.session_state["cons_candidates"] = [
            {
                "classification": "SEMANTIC",
                "content": "",
                "importance": 0.7,
                "confidence": 0.9,
                "entities": [],
                "source": "conversation",
                "reason": "",
            }
        ]

    st.markdown("**Candidates**")
    for i, cand in enumerate(list(st.session_state["cons_candidates"])):
        with st.container(border=True):
            c1, c2 = st.columns([1, 3])
            cand["classification"] = c1.selectbox(
                "Classification",
                options=["SEMANTIC", "EPISODIC", "PROCEDURAL", "IGNORE"],
                index=["SEMANTIC", "EPISODIC", "PROCEDURAL", "IGNORE"].index(
                    cand.get("classification", "SEMANTIC")
                ),
                key=f"cons-class-{i}",
            )
            cand["content"] = c2.text_input(
                "Content", value=cand.get("content", ""), key=f"cons-content-{i}"
            )
            c1, c2, c3 = st.columns(3)
            cand["importance"] = c1.slider(
                "Importance",
                0.0,
                1.0,
                float(cand.get("importance", 0.7)),
                key=f"cons-imp-{i}",
            )
            cand["confidence"] = c2.slider(
                "Confidence",
                0.0,
                1.0,
                float(cand.get("confidence", 0.9)),
                key=f"cons-conf-{i}",
            )
            cand["reason"] = c3.text_input(
                "Reason", value=cand.get("reason", ""), key=f"cons-reason-{i}"
            )

    c1, c2 = st.columns(2)
    if c1.button("Add candidate", key="btn-add-cons"):
        st.session_state["cons_candidates"].append(
            {
                "classification": "SEMANTIC",
                "content": "",
                "importance": 0.7,
                "confidence": 0.9,
                "entities": [],
                "source": "conversation",
                "reason": "",
            }
        )
        st.rerun()
    if c2.button("Reset", key="btn-reset-cons"):
        st.session_state["cons_candidates"] = [
            {
                "classification": "SEMANTIC",
                "content": "",
                "importance": 0.7,
                "confidence": 0.9,
                "entities": [],
                "source": "conversation",
                "reason": "",
            }
        ]
        st.rerun()

    if st.button("Run consolidation", key="btn-run-cons", type="primary"):
        candidates_filtered = [
            c for c in st.session_state["cons_candidates"] if c.get("content", "").strip()
        ]
        if not candidates_filtered:
            st.warning("Add at least one candidate with content.")
        elif not user_msg.strip():
            st.warning("User message is required.")
        else:
            body = {
                "user_message": user_msg,
                "assistant_reply": assistant_msg,
                "conversation_id": conv_id,
                "candidates": candidates_filtered,
            }
            with st.spinner("Consolidating..."):
                ok, data = _post("/memory/consolidate", json=body, timeout=90.0)
            if not ok:
                st.error(f"Consolidation failed: {data.get('error')}")
            else:
                st.success(f"Source turn memory: `{data['source_memory_id']}`")
                outcomes = data.get("outcomes", [])
                if outcomes:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "idx": o["candidate_index"],
                                    "class": o["classification"],
                                    "action": o["action"],
                                    "new_id": o.get("created_memory_id") or "-",
                                    "conflicts": len(o.get("conflicts", [])),
                                }
                                for o in outcomes
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.json(data, expanded=False)
