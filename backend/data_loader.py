"""Raw-file loading. Files are whitespace separated, headerless (see Dataset doc)."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (n,3) lon, lat, t
    nodes: np.ndarray            # (N,2) lon, lat ; row index = node id
    arcs: np.ndarray             # (M,2) from_node, to_node ; row index = arc id
    route: np.ndarray | None     # reference arc ids (SERVER-SIDE ONLY, never sent to UI/model)


class TripStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)

    def ids(self) -> list[str]:
        need = (".track", ".nodes", ".arcs")
        return sorted(
            p.stem for p in self.dir.glob("*.track")
            if all((self.dir / (p.stem + e)).exists() for e in need)
        )

    def load(self, trip_id: str) -> Trip:
        if trip_id not in self.ids():            # also blocks path traversal
            raise KeyError(trip_id)
        return self._load(trip_id)

    @lru_cache(maxsize=4)
    def _load(self, trip_id: str) -> Trip:
        d = self.dir
        track = np.loadtxt(d / f"{trip_id}.track", ndmin=2)
        nodes = np.loadtxt(d / f"{trip_id}.nodes", ndmin=2)
        arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
        rp = d / f"{trip_id}.route"
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1) if rp.exists() else None
        return Trip(trip_id, track, nodes, arcs, route)
