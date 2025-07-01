# Minimal test for DragModel (place in the same directory)
from v344 import DragModel

def test_g1_drag():
    assert 0.25 < DragModel.G1(100) < 0.35