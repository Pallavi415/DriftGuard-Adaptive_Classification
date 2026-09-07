# Import NumPy for creating test signals
import numpy as np

# Import the Page-Hinkley detector
from driftaware.drift.page_hinkley import PageHinkley


# Test that the detector identifies a persistent increase in a signal
def test_page_hinkley_detects_change():
    # Create a detector with a lower threshold for a small test signal
    detector = PageHinkley(
        delta=0.01,
        threshold=5.0
    )

    # Create a stable signal followed by a sustained increase
    stable_values = np.zeros(50)
    changed_values = np.ones(50)

    # Track whether a change is detected
    detected = False

    # Process the stable part of the signal
    for value in stable_values:
        if detector.update(value):
            detected = True
            break

    # Process the changed part of the signal
    if not detected:
        for value in changed_values:
            if detector.update(value):
                detected = True
                break

    # Confirm that the persistent change was detected
    assert detected