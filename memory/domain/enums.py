"""Enums for memory types and relationship kinds. String values match OpenSearch keyword content."""

from __future__ import annotations

from enum import StrEnum


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class RelationshipType(StrEnum):
    DERIVED_FROM = "DERIVED_FROM"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CAUSED_BY = "CAUSED_BY"
    RESULTED_IN = "RESULTED_IN"
    USED_BY = "USED_BY"
    PRECEDES = "PRECEDES"
    REINFORCES = "REINFORCES"
    SUPERSEDES = "SUPERSEDES"
