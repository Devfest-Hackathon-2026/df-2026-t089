"""Convert the SADIE II KU100 HRIRs to the small file the app convolves sounds with.

  pip install h5py numpy
  # https://zenodo.org/records/12092466/files/D1_HRIR_SOFA.zip -> D1_44K_16bit_256tap_FIR_SOFA.sofa
  python tools/export_hrtf.py D1_44K_16bit_256tap_FIR_SOFA.sofa

Writes app/src/main/assets/hrtf/sadie_d1_ku100.bin, little-endian:
  int32 count, int32 taps, float32 (azimuth, elevation) * count in degrees (SOFA: 0 is front, azimuth counter-clockwise),
  int16 ir[count][2 ears, left first][taps], scaled by 32767.
"""
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "app/src/main/assets/hrtf/sadie_d1_ku100.bin"
MIN_SPACING_DEG = 4.0  # 8802 measured directions would add 9 MB; this spacing keeps about a third


def unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)


def thin(az, el, spacing_deg):
    """Greedy thinning that keeps the horizontal plane first, because the app's sounds are near ear height."""
    v = unit(az, el)
    limit = np.cos(np.radians(spacing_deg))
    free = np.ones(len(az), bool)
    kept = []
    for i in np.lexsort((az, np.abs(el))):
        if free[i]:
            kept.append(i)
            free &= v @ v[i] < limit
    return np.array(sorted(kept))


def main():
    with h5py.File(sys.argv[1], "r") as f:
        ir, rate, pos = f["Data.IR"][()], float(np.ravel(f["Data.SamplingRate"][()])[0]), f["SourcePosition"][()]
    assert rate == 44100 and ir.shape[1:] == (2, 256), (rate, ir.shape)
    az, el = pos[:, 0], pos[:, 1]
    left = np.argmin(np.abs(az - 90) + np.abs(el))
    assert (ir[left, 0] ** 2).sum() > (ir[left, 1] ** 2).sum(), "expected the left ear first"
    assert np.abs(ir).max() <= 1
    k = thin(az, el, MIN_SPACING_DEG)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("wb") as out:
        out.write(struct.pack("<ii", len(k), ir.shape[2]))
        out.write(np.stack([az[k], el[k]], 1).astype("<f4").tobytes())
        out.write(np.round(ir[k] * 32767).astype("<i2").tobytes())
    print(f"{len(k)} of {len(az)} directions, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
