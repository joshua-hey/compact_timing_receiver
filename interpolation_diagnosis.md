# Interpolation Diagnosis

Seed: `100`

Estimator modes compared: `none` and `parabolic` matched-filter interpolation.

High-SNR point: `30.0` dB input SNR

High-SNR RMSE before interpolation: `0.278997` samples
High-SNR RMSE after interpolation: `0.00857369` samples
High-SNR CRLB sigma: `0.00670906` samples
High-SNR efficiency before interpolation: `41.5851`
High-SNR efficiency after interpolation: `1.27793`
Improvement factor: `32.5411`

Conclusion: Parabolic interpolation removes most of the observed high-SNR sample-grid floor and measures 1.28x the CRLB at 30.0 dB.
