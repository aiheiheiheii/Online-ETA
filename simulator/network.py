"""Reproducible edge topology and current-state minimum-cost routing.

Links are directed tuples ``(source_node, destination_node)``.  The default
ring-and-chord topology creates both directions of each physical connection,
so each direction has its own FIFO backlog and effective bandwidth.

For tractable repeated online decisions, the *available* path set for a node
pair is the first ``max_candidate_paths`` simple paths ordered by hop count
in the fixed topology (and optionally limited by ``max_path_hops``).  These
structural paths are cached.  At every decision, the route minimizes the
paper's Eq. (10), sum((current_link_backlog + activation_mb) / current_rate),
over that configured path set.  No future resource state enters path choice.
"""

from __future__ import annotations

from math import isfinite
from random import Random
from typing import Iterable, Mapping, Protocol

import networkx as nx


LinkId = tuple[int, int]
Path = tuple[LinkId, ...]


class TopologySettings(Protocol):
    """Configuration fields consumed by the topology generator."""

    num_nodes: int
    topology_model: str
    ring_neighbors: int
    extra_chords_per_node: int
    max_path_hops: int | None
    max_candidate_paths: int


class Network:
    """Fixed directed graph with cached structural route candidates."""

    def __init__(
        self,
        nodes: Iterable[int],
        links: Iterable[LinkId],
        *,
        max_path_hops: int | None = None,
        max_candidate_paths: int = 3,
    ) -> None:
        self.nodes = tuple(sorted(set(nodes)))
        if not self.nodes:
            raise ValueError("A network needs at least one node")
        self.links = tuple(sorted(set(links)))
        if max_path_hops is not None and max_path_hops <= 0:
            raise ValueError("max_path_hops must be positive or None")
        if max_candidate_paths <= 0:
            raise ValueError("max_candidate_paths must be positive")
        node_set = set(self.nodes)
        for src, dst in self.links:
            if src not in node_set or dst not in node_set or src == dst:
                raise ValueError(f"Invalid directed link {(src, dst)}")
        self.max_path_hops = max_path_hops
        self.max_candidate_paths = max_candidate_paths
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(self.nodes)
        self._graph.add_edges_from(self.links)
        self._path_cache: dict[tuple[int, int], tuple[Path, ...]] = {}

    def _check_nodes(self, src: int, dst: int) -> None:
        if src not in self._graph or dst not in self._graph:
            raise ValueError(f"Unknown node in route {src}->{dst}")

    def has_path(self, src: int, dst: int) -> bool:
        """Whether fixed topology admits a path within the configured hop cap."""

        self._check_nodes(src, dst)
        if src == dst:
            return True
        return bool(self.candidate_paths(src, dst))

    def candidate_paths(self, src: int, dst: int) -> tuple[Path, ...]:
        """Return the cached, topology-only available path set."""

        self._check_nodes(src, dst)
        if src == dst:
            return ((),)
        key = (src, dst)
        if key not in self._path_cache:
            paths: list[Path] = []
            try:
                generator = nx.shortest_simple_paths(self._graph, src, dst)
                for node_path in generator:
                    hops = len(node_path) - 1
                    if self.max_path_hops is not None and hops > self.max_path_hops:
                        break
                    paths.append(tuple(zip(node_path[:-1], node_path[1:])))
                    if len(paths) >= self.max_candidate_paths:
                        break
            except nx.NetworkXNoPath:
                pass
            self._path_cache[key] = tuple(paths)
        return self._path_cache[key]

    def best_path(
        self,
        src: int,
        dst: int,
        data_mb: float,
        backlog: Mapping[LinkId, float],
        rates: Mapping[LinkId, float],
    ) -> tuple[Path, float]:
        """Choose the available path with minimum current Eq. (10) cost.

        ``backlog`` and ``data_mb`` are MB; ``rates`` are MB/s.  Equal costs
        use fewer hops and then lexicographic link IDs as deterministic ties.
        """

        if not isfinite(data_mb) or data_mb < 0:
            raise ValueError("data_mb must be finite and nonnegative")
        paths = self.candidate_paths(src, dst)
        if not paths:
            raise ValueError(f"No available path from node {src} to {dst}")
        best: tuple[float, int, Path] | None = None
        for path in paths:
            cost = 0.0
            for link_id in path:
                if link_id not in rates:
                    raise ValueError(f"Missing bandwidth for link {link_id}")
                rate = rates[link_id]
                queued = backlog.get(link_id, 0.0)
                if not isfinite(rate) or rate <= 0:
                    raise ValueError(f"Invalid bandwidth for link {link_id}: {rate}")
                if not isfinite(queued) or queued < 0:
                    raise ValueError(f"Invalid backlog for link {link_id}: {queued}")
                cost += (queued + data_mb) / rate
            candidate = (cost, len(path), path)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        return best[2], best[0]


def build_network(config: TopologySettings, seed: int) -> Network:
    """Generate one reproducible topology shared by every method in a replicate."""

    if config.num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    nodes = tuple(range(config.num_nodes))
    undirected: set[tuple[int, int]] = set()
    if config.topology_model == "complete":
        undirected = {
            (src, dst)
            for src in nodes
            for dst in nodes
            if src < dst
        }
    elif config.topology_model == "ring_chord":
        if config.num_nodes > 1 and config.ring_neighbors < 1:
            raise ValueError("ring_chord needs ring_neighbors >= 1 to stay connected")
        if config.ring_neighbors < 0 or config.extra_chords_per_node < 0:
            raise ValueError("ring_neighbors and extra_chords_per_node must be nonnegative")
        for src in nodes:
            for distance in range(1, min(config.ring_neighbors, config.num_nodes - 1) + 1):
                dst = (src + distance) % config.num_nodes
                undirected.add((min(src, dst), max(src, dst)))
        rng = Random(seed)
        for src in nodes:
            for _ in range(config.extra_chords_per_node):
                choices = [
                    dst for dst in nodes
                    if dst != src and (min(src, dst), max(src, dst)) not in undirected
                ]
                if not choices:
                    break
                dst = rng.choice(choices)
                undirected.add((min(src, dst), max(src, dst)))
    else:
        raise ValueError(f"Unsupported topology_model: {config.topology_model}")

    directed = tuple(
        link
        for src, dst in sorted(undirected)
        for link in ((src, dst), (dst, src))
    )
    network = Network(
        nodes,
        directed,
        max_path_hops=config.max_path_hops,
        max_candidate_paths=config.max_candidate_paths,
    )
    if len(nodes) > 1 and not nx.is_strongly_connected(network._graph):
        raise AssertionError("Generated topology is not strongly connected")
    return network
