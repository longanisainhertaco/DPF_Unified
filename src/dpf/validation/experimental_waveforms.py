"""Digitized and reconstructed current waveform arrays for DPF devices.

Each array pair (_*_T_US, _*_I_*) holds the raw point data before unit
conversion.  Device objects in experimental_devices.py apply the unit
conversions (us->s, kA->A, MA->A) when constructing ExperimentalDevice
instances.
"""

from __future__ import annotations

import numpy as np

# =====================================================================
# PF-1000 at 27 kV — Scholz et al., Nukleonika 51(1), 2006, Fig. 2
# 26 points covering 0-10 us, interpolated from published waveform
# Characteristic features: rise to ~1.87 MA at ~5.8 us, current dip at ~7 us
# =====================================================================

_PF1000_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.3, 5.6, 5.8, 6.0, 6.3, 6.5, 6.8, 7.0, 7.3,
    7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
])
_PF1000_WAVEFORM_I_MA = np.array([
    0.00, 0.15, 0.35, 0.58, 0.82, 1.05, 1.25, 1.42, 1.56, 1.67,
    1.76, 1.81, 1.85, 1.87, 1.86, 1.82, 1.75, 1.55, 1.40, 1.30,
    1.25, 1.15, 1.05, 0.95, 0.85, 0.75,
])

# =====================================================================
# UNU-ICTP PFF measured I(t) from IPFS "UNU ICTPPFF D2 05.15.xls"
# 45 points covering 0-5 us at 13.5 kV, 3.0 Torr D2
# Median-filtered to remove EMI spike at pinch (~2.72-2.73 us)
# Characteristic features: rise to ~169 kA at ~2.2-2.6 us, shallow 14% dip at ~2.76 us
# =====================================================================

_UNU_ICTP_WAVEFORM_T_US = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.65, 2.70, 2.73,
    2.76, 2.80, 2.85, 2.90, 2.95, 3.0, 3.1, 3.2, 3.3, 3.5,
    3.7, 4.0, 4.3, 4.5, 5.0,
])
_UNU_ICTP_WAVEFORM_I_KA = np.array([
    8.7, 18.8, 28.1, 40.6, 56.3, 65.6, 73.8, 84.4, 93.8, 103.1,
    112.5, 112.5, 121.9, 131.3, 140.6, 140.6, 150.0, 150.0, 159.4, 159.4,
    159.4, 161.9, 168.8, 168.8, 168.8, 168.8, 168.8, 164.4, 159.4, 151.3,
    145.0, 153.8, 155.0, 150.0, 150.0, 148.8, 150.0, 140.6, 140.6, 131.3,
    121.9, 112.5, 103.1, 93.8, 63.1,
])

# =====================================================================
# SYNTHETIC -- DO NOT TREAT AS MEASUREMENT
# PF-1000 at 16 kV -- not digitized from a published figure.
# Source paper: Akel et al., Radiat. Phys. Chem. 188:109633, 2021
# Same device (IPPLM Warsaw), different operating conditions:
#   V0 = 16 kV (vs 27 kV), fill pressure = 1.05 Torr D2 (vs 3.5 Torr)
# Reconstructed from physics scaling of 27 kV Scholz waveform, rescaled to
# Akel's published peak I_peak = 1.165 MA (shot 12581, Table 1: Ipeak = 1165 kA).
# Previous version used I_peak = 1.20 MA (3% too high, no paper match).
# Array rescaled by 1.165/1.20 = 0.9708333 to match Akel Table 1 verbatim.
# Replace with actual digitized data from Akel (2021) Fig. 3 if/when
# available; until then, do not use this array as a validation target
# for the *shape* of the waveform -- only as an order-of-magnitude
# constraint on the peak.
# =====================================================================

_SYNTHETIC_PF1000_16KV_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.3, 5.5, 5.7, 5.8, 6.0, 6.3, 6.5, 7.0, 7.5,
    8.0, 8.5, 9.0, 9.5, 10.0,
])
_SYNTHETIC_PF1000_16KV_WAVEFORM_I_MA = np.array([
    0.0000, 0.0971, 0.2233, 0.3689, 0.5243, 0.6699, 0.7961, 0.9029, 0.9902, 1.0679,
    1.1262, 1.1553, 1.1650, 1.1456, 1.0873, 0.9708, 0.8738, 0.8252, 0.7573, 0.6990,
    0.6408, 0.5825, 0.5243, 0.4757, 0.4272,
])

# =====================================================================
# PF-1000 at 27 kV — Gribkov et al., J. Phys. D: Appl. Phys. 40:1977-1989 (Part I), 2007
# doi:10.1088/0022-3727/40/7/021
# (The old citation "40:3592" was Scholz et al., Part II, not Gribkov Part I.)
# WAVEFORM PROVENANCE: Data read from plasmafocus.net/IPFS/PF1000 05.15.xls Sheet2,
# NOT from the Gribkov paper directly. The paper itself (Gribkov 2007 Part I)
# does not publish a tabulated I(t) trace; the xls file is the authoritative
# digital archive distributed by the Lee RADPF model developers.
# Same device and operating conditions as Scholz (2006), but DIFFERENT shot
# and DIFFERENT digitization source. Peak 1.846 MA at 6.39 us.
# 94 data points (vs 26 for Scholz), covering -1.68 to 14.73 us.
# Source: plasmafocus.net/IPFS/machines/PF1000 05.15.xls, Sheet2
# =====================================================================

_PF1000_GRIBKOV_WAVEFORM_T_US = np.array([
    -1.682, -1.169, -0.599, -0.285, 0.000, 0.085, 0.141, 0.198, 0.310,
    0.367, 0.592, 0.648, 0.732, 0.845, 0.930, 1.072, 1.099, 1.213,
    1.354, 1.496, 1.581, 1.638, 1.837, 2.064, 2.262, 2.518, 2.660,
    2.745, 2.859, 3.143, 3.342, 3.428, 3.569, 3.712, 3.911, 4.139,
    4.338, 4.509, 4.652, 4.822, 4.908, 5.193, 5.421, 5.677, 5.905,
    6.133, 6.390, 6.590, 6.818, 6.932, 7.103, 7.331, 7.445, 7.702,
    7.874, 8.017, 8.246, 8.390, 8.477, 8.563, 8.593, 8.679, 8.737,
    8.881, 8.995, 9.109, 9.280, 9.423, 9.536, 9.651, 9.965, 10.222,
    10.365, 10.622, 10.907, 11.136, 11.136, 11.335, 11.564, 11.564,
    11.792, 12.049, 12.049, 12.534, 12.705, 12.934, 13.105, 13.362,
    13.562, 13.819, 14.019, 14.304, 14.532, 14.732,
])
_PF1000_GRIBKOV_WAVEFORM_I_KA = np.array([
    -12.188, -22.772, -11.396, -16.646, -10.959, 49.377, 93.254,
    148.090, 235.843, 290.679, 504.542, 581.295, 652.590, 707.467,
    762.323, 844.619, 899.433, 954.311, 1031.130, 1102.460, 1129.920,
    1157.360, 1201.340, 1267.260, 1349.600, 1382.660, 1421.120,
    1475.980, 1503.460, 1596.820, 1591.480, 1618.940, 1662.880,
    1701.340, 1706.970, 1734.530, 1751.110, 1784.120, 1767.780,
    1811.740, 1811.800, 1822.970, 1817.660, 1839.760, 1845.410,
    1845.580, 1845.760, 1840.430, 1840.600, 1840.000, 1829.600,
    1835.490, 1830.100, 1820.000, 1790.000, 1748.320, 1655.340,
    1584.210, 1507.560, 1430.910, 1370.660, 1315.930, 1261.180,
    1173.610, 1160.000, 1173.780, 1168.420, 1146.610, 1171.000,
    1135.820, 1130.570, 1125.280, 1081.550, 1076.250, 1049.060,
    1032.790, 1032.790, 1032.940, 1016.670, 1016.670, 1000.400,
    989.625, 989.625, 940.664, 935.310, 908.080, 897.246, 875.516,
    859.223, 848.452, 821.201, 810.450, 794.179, 794.325,
])

# Trim to t >= 0 for consistency with other waveforms
_gribkov_mask = _PF1000_GRIBKOV_WAVEFORM_T_US >= 0.0
PF1000_GRIBKOV_T_TRIMMED = _PF1000_GRIBKOV_WAVEFORM_T_US[_gribkov_mask]
PF1000_GRIBKOV_I_TRIMMED = _PF1000_GRIBKOV_WAVEFORM_I_KA[_gribkov_mask]

# =====================================================================
# POSEIDON at 60 kV — IPFS (plasmafocus.net) digitized I(t) waveform
# Different bank configuration from POSEIDON (40 kV): C=156 uF, V=60 kV
# 35 subsampled points from 103-point digitized waveform
# =====================================================================

_POSEIDON60KV_WAVEFORM_T_US = np.array([
    0.007, 0.092, 0.148, 0.205, 0.261, 0.339, 0.395, 0.452, 0.530, 0.608,
    0.686, 0.764, 0.849, 0.927, 1.027, 1.141, 1.262, 1.405, 1.577, 1.770,
    1.978, 2.186, 2.394, 2.537, 2.603, 2.675, 2.734, 2.814, 2.929, 3.123,
    3.281, 3.439, 3.619, 3.770, 3.914,
])
_POSEIDON60KV_WAVEFORM_I_KA = np.array([
    0, 267, 499, 697, 918, 1130, 1290, 1460, 1660, 1850,
    2010, 2170, 2330, 2460, 2620, 2760, 2890, 3010, 3110, 3170,
    3190, 3180, 3150, 3050, 2890, 2680, 2490, 2280, 2140, 2100,
    1990, 1890, 1800, 1700, 1580,
])

# =====================================================================
# SYNTHETIC -- DO NOT TREAT AS MEASUREMENT
# FAETON-I (Fuse Energy) -- 100 kV, 125 kJ, ~1 MA dense plasma focus.
# Source paper: Damideh et al., Sci. Rep. 15:23048, 2025
# Not digitized from a published figure. Reconstructed from damped RLC
# (C=25 uF, L=220 nH, R=7.6 mOhm).
# Time axis shifted +0.3 us (peak: 3.4 us -> 3.7 us) to match Damideh 2025
# §III: "generates ~1 MA of electrical current with a rise time of ~3.7 us"
# (pp.3,4) and transition time 3.745 us (p.9, radial-phase trajectory).
# Replace with digitized data from Damideh (2025) Fig. 3 when available.
# Until then, do not use this array as a validation target for waveform
# *shape* -- only as a placeholder for circuit-level checks.
# =====================================================================

_SYNTHETIC_FAETON_WAVEFORM_T_US = np.array([
    0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0,
    3.3, 3.5, 3.7, 3.9, 4.0, 4.1, 4.3, 4.5, 4.8, 5.3,
    5.8, 6.3, 6.8, 7.3, 7.7,
])
_SYNTHETIC_FAETON_WAVEFORM_I_KA = np.array([
    0.0, 135.3, 267.0, 393.0, 511.3, 620.1, 717.6, 802.5, 873.5, 929.6,
    969.9, 987.9, 998.5, 991.4, 983.7, 973.1, 949.1, 932.3, 913.7, 829.1,
    694.8, 531.4, 346.8, 149.9, -10.5,
])

# =====================================================================
# SYNTHETIC -- DO NOT TREAT AS MEASUREMENT
# MJOLNIR (LLNL) -- 2 MJ MA-class deuterium DPF at 60 kV typical
# operation. Not digitized from a published figure. Reconstructed from
# a known peak current (2.8 MA), rise time (~5 us), and estimated
# circuit parameters; the dip/recovery section is illustrative, not
# measured.
# Replace with digitized data from Schmidt (2021) or Goyon (2025) when
# available. Until then, do not use this array as a validation target
# for waveform *shape* -- only as a placeholder for circuit-level
# checks.
# =====================================================================

_SYNTHETIC_MJOLNIR_WAVEFORM_T_US = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.3,
    4.5, 4.7, 5.0, 5.2, 5.5, 5.8, 6.0, 6.5, 7.0, 7.5,
    8.0, 8.5, 9.0, 9.5, 10.0,
])
_SYNTHETIC_MJOLNIR_WAVEFORM_I_KA = np.array([
    0, 438, 865, 1271, 1646, 1980, 2265, 2495, 2663, 2733,
    2766, 2788, 2800, 2554, 2184, 2318, 2408, 2329, 2253, 2179,
    2107, 2038, 1972, 1907, 1844,
])
