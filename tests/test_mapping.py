"""mapping.assign on synthetic masks and boxes (CPU, offline, no model weights, no data/raw)."""

from __future__ import annotations

import numpy as np
import pytest

from avtopassport.pipeline.mapping import MIN_OVERLAP, UNKNOWN_PART, assign, overlap
from avtopassport.schema import Segment

H, W = 100, 200


def rect_seg(cls: str, x1: int, y1: int, x2: int, y2: int, *, score: float = 0.9, mask: bool = True) -> Segment:
    """Rectangle segment; with mask=False it is detection-only (area_px = box area)."""
    box = (float(x1), float(y1), float(x2), float(y2))
    if not mask:
        return Segment(cls, score, box, None, (x2 - x1) * (y2 - y1))
    m = np.zeros((H, W), bool)
    m[y1:y2, x1:x2] = True
    return Segment(cls, score, box, m, int(m.sum()))


def test_largest_overlap_part_wins() -> None:
    door = rect_seg("left_front-door", 0, 0, 100, 100)
    fender = rect_seg("left_fender", 100, 0, 200, 100)
    dent = rect_seg("dent", 90, 40, 120, 50)   # 10 px wide in door, 20 px wide in fender
    (obs,) = assign("CAR1", [dent], [door, fender])
    assert obs.part == "left_fender"
    assert obs.part_overlap == pytest.approx(20 / 30)
    assert obs.source == "CAR1" and obs.damage == "dent" and obs.score == pytest.approx(0.9)
    assert obs.box == (90.0, 40.0, 120.0, 50.0)


def test_area_ratio_is_damage_over_part() -> None:
    door = rect_seg("left_front-door", 0, 0, 100, 100)          # 10000 px
    scratch = rect_seg("scratch", 10, 10, 30, 30)               # 400 px
    (obs,) = assign("f0", [scratch], [door])
    assert obs.damage_area_px == 400 and obs.part_area_px == 10000
    assert obs.area_ratio == pytest.approx(0.04)
    assert obs.part_overlap == pytest.approx(1.0)


def test_unknown_when_no_overlap() -> None:
    door = rect_seg("left_front-door", 0, 0, 50, 50)
    dent = rect_seg("dent", 150, 60, 170, 80)
    (obs,) = assign("f0", [dent], [door])
    assert obs.part == UNKNOWN_PART
    assert obs.part_area_px == 0 and obs.area_ratio == 0.0 and obs.part_overlap == 0.0
    assert obs.damage_area_px == 400


def test_unknown_when_no_parts_at_all() -> None:
    (obs,) = assign("f0", [rect_seg("crack", 0, 0, 10, 10)], [])
    assert obs.part == UNKNOWN_PART and obs.part_overlap == 0.0 and obs.area_ratio == 0.0


def test_min_overlap_threshold() -> None:
    door = rect_seg("hood", 0, 0, 100, 100)
    dent = rect_seg("dent", 90, 0, 140, 10)   # 10 of 50 columns inside -> overlap 0.2
    (at,) = assign("f0", [dent], [door])
    assert at.part_overlap == pytest.approx(0.2) and MIN_OVERLAP == 0.2
    assert at.part == "hood"                                    # >= min_overlap is kept
    (below,) = assign("f0", [dent], [door], min_overlap=0.21)
    assert below.part == UNKNOWN_PART and below.part_overlap == pytest.approx(0.2)


def test_box_fallback_when_a_mask_is_missing() -> None:
    door_box = rect_seg("left_front-door", 0, 0, 100, 100, mask=False)
    fender_box = rect_seg("left_fender", 100, 0, 200, 100, mask=False)
    dent_masked = rect_seg("dent", 80, 0, 110, 10)   # 20 of 30 columns in the door box
    (obs,) = assign("f0", [dent_masked], [door_box, fender_box])
    assert obs.part == "left_front-door"
    assert obs.part_overlap == pytest.approx(20 / 30)
    assert obs.area_ratio == pytest.approx(300 / 10000)

    door_mask = rect_seg("left_front-door", 0, 0, 100, 100)
    dent_box = rect_seg("dent", 95, 0, 105, 10, mask=False)
    (obs2,) = assign("f0", [dent_box], [door_mask])
    assert obs2.part == "left_front-door" and obs2.part_overlap == pytest.approx(0.5)


def test_mask_overlap_uses_pixels_not_boxes() -> None:
    """An L-shaped part whose box covers the damage but whose mask does not."""
    m = np.zeros((H, W), bool)
    m[:, :20] = True
    m[80:, :] = True
    part = Segment("bumper", 0.9, (0.0, 0.0, float(W), float(H)), m, int(m.sum()))
    dent = rect_seg("dent", 100, 10, 120, 30)
    assert overlap(dent, part) == 0.0
    assert assign("f0", [dent], [part])[0].part == UNKNOWN_PART


def test_tie_goes_to_first_part() -> None:
    a = rect_seg("left_front-door", 0, 0, 100, 100)
    b = rect_seg("left_rear-door", 100, 0, 200, 100)
    dent = rect_seg("dent", 90, 0, 110, 10)
    assert assign("f0", [dent], [a, b])[0].part == "left_front-door"
    assert assign("f0", [dent], [b, a])[0].part == "left_rear-door"


def test_license_plate_is_a_normal_part() -> None:
    plate = rect_seg("license-plate", 50, 50, 150, 80)
    bumper = rect_seg("front_bumper", 0, 40, 200, 100)
    crack = rect_seg("crack", 60, 55, 70, 65)
    (obs,) = assign("f0", [crack], [plate, bumper])
    assert obs.part == "license-plate"


def test_several_damages_keep_input_order() -> None:
    door = rect_seg("hood", 0, 0, 200, 100)
    ds = [rect_seg("scratch", 0, 0, 5, 5), rect_seg("dent", 10, 10, 20, 20), rect_seg("crack", 30, 30, 40, 40)]
    assert [o.damage for o in assign("f0", ds, [door])] == ["scratch", "dent", "crack"]
    assert assign("f0", [], [door]) == []


def test_empty_damage_mask_is_unknown() -> None:
    door = rect_seg("hood", 0, 0, 200, 100)
    empty = Segment("dent", 0.5, (0.0, 0.0, 10.0, 10.0), np.zeros((H, W), bool), 0)
    (obs,) = assign("f0", [empty], [door])
    assert obs.part == UNKNOWN_PART and obs.part_overlap == 0.0


def test_mask_shape_mismatch_raises() -> None:
    door = rect_seg("hood", 0, 0, 100, 100)
    small = Segment("dent", 0.5, (0.0, 0.0, 10.0, 10.0), np.ones((50, 50), bool), 2500)
    with pytest.raises(ValueError, match="shape"):
        assign("f0", [small], [door])
