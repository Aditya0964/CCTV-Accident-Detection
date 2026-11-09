"""Pair-level event accumulator and one-candidate-per-encounter reporting."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PairEvent:
    ids: tuple[int, int]
    first_contact: float
    last_contact: float
    region: list[float]
    speed_drop: bool = False
    direction_change: bool = False
    close: bool = False
    overlap: bool = False
    post_stop: bool = False
    emitted: bool = False
    details: dict = field(default_factory=dict)

    def score(self) -> int:
        return 2*int(self.speed_drop) + 2*int(self.direction_change) + 2*int(self.close) + int(self.overlap) + 2*int(self.post_stop)

    def triggered_rules(self) -> list[str]:
        return [name for name, flag in [("speed_drop",self.speed_drop), ("direction_change",self.direction_change),
                                       ("proximity",self.close), ("overlap",self.overlap), ("post_stop",self.post_stop)] if flag]

    def candidate(self, decision_time: float) -> dict:
        return {"candidate_timestamp_sec": round(self.first_contact,6),
                "decision_timestamp_sec": round(decision_time,6),
                "vehicle_ids": list(self.ids), "anomaly_score": self.score(),
                "incident_region_xyxy": [round(x,2) for x in self.region],
                "triggered_rules": self.triggered_rules()}
