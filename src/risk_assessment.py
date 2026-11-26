"""Assign a visual risk label from the evidence available for an incident."""


def compute_visual_risk(
    *, fire: bool | None, smoke: bool | None, motion_score: int | None, num_vehicles: int | None,
) -> tuple[str, str]:
    if fire is True:
        return "High", "Persistent fire was detected."
    if smoke is True:
        return "Elevated", "Persistent smoke was detected."
    motion_known = type(motion_score) is int and 0 <= motion_score <= 9
    vehicles_known = type(num_vehicles) is int and num_vehicles >= 1
    if motion_known and vehicles_known and motion_score >= 7 and num_vehicles >= 2:
        return "Elevated", "Motion score is at least 7/9 and two or more vehicles were involved."
    if fire is not False or smoke is not False or not motion_known or not vehicles_known:
        return "Unknown", "The hazard scan or incident motion and vehicle evidence is incomplete."
    return "Low Observed", "The completed scan found no persistent hazard or strong multi-vehicle motion evidence."
