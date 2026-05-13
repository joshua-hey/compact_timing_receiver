# Floor Diagnosis

Seed: `100`

Beta RMS: `562.63` Hz, computed numerically from the sampled Gaussian pulse template used by the simulator.

SNR convention: the sweep input SNR is full-waveform average signal power divided by AWGN sample noise power. The CRLB overlay uses the estimated post-correlation peak SNR from the matched-filter response.

Detection threshold rule: fixed matched-filter correlation-height threshold `0.2` with `find_peaks` distance `0.01` seconds. This is not CFAR and is not a fixed-Pfa detector.

Search window length: `20200` samples. Resolution cells per trial: `202`.

False detections are reported as an empirical extra-detection rate per resolution cell or per 100 true pulses, not as formal Pfa.

## SNR Convention Check

| input_snr_db | post_correlation_snr_db | processing_gain_db |
| ---: | ---: | ---: |
| 30.0 | 49.61 | 19.61 |
| 25.0 | 44.60 | 19.60 |
| 20.0 | 39.54 | 19.54 |
| 15.0 | 34.62 | 19.62 |
| 10.0 | 29.62 | 19.62 |
| 8.0 | 27.58 | 19.58 |
| 7.5 | 27.13 | 19.63 |
| 7.0 | 26.55 | 19.55 |
| 6.5 | 25.87 | 19.37 |
| 6.0 | 25.36 | 19.36 |
| 5.5 | 25.04 | 19.54 |
| 5.0 | 24.53 | 19.53 |
| 4.0 | 23.53 | 19.53 |
| 3.0 | 22.59 | 19.59 |
| 2.0 | 21.55 | 19.55 |
| 1.0 | 20.64 | 19.64 |
| 0.0 | 19.53 | 19.53 |

## Floor Tests

| hypothesis | parameter changed | RMSE before samples | RMSE after samples | conclusion |
| --- | --- | ---: | ---: | --- |
| H1 | diagnostic peak interpolation grid 10x | 0.2151 | 0.03651 | RMSE dropped with finer diagnostic interpolation. |
| H1 | diagnostic peak interpolation grid 100x | 0.2151 | 0.008479 | RMSE dropped with finer diagnostic interpolation. |
| H2 | SNR convention only | 0.2151 | 0.2151 | This changes CRLB scaling, not estimator RMSE. |

Conclusion: The high-SNR floor is dominated by sample-grid peak picking.
