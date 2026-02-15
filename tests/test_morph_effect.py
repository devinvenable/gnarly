import numpy as np
from PIL import Image

from gnarly.effects.morph_effect import MorphEffect, MorphState


def test_morph_state_blend_progression():
    state = MorphState(
        obj_id="obj",
        region=(0, 0, 10, 10),
        generated_image=Image.new("RGB", (10, 10), color="white"),
        frame_start=0,
        frames=10,
        min_blend=0.1,
        max_blend=0.7,
    )

    assert state.get_blend_factor() == 0.1
    state.update()
    assert round(state.get_blend_factor(), 3) == 0.16


def test_morph_effect_no_detector_is_noop():
    effect = MorphEffect(detector=None)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    output = effect.apply(frame)
    assert np.array_equal(output, frame)
