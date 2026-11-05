HYPERNET-W BERING CLOUD PILOT

This is a private two-file portability/throughput pilot, not an accuracy test.
HyperNet-W v3 has 1,566,756 parameters and no borrowed neural weights. It was
pretrained on 77,000 REAL unlabeled acoustic windows, not mathematical shapes.
One fixed positive support per target does not remove the earlier labeled
checkpoint/depth selection budget. See protocol.json and overlap_audit.json.

Allowed search deployments are ONLY Bering01-m2 and Bering04-m4. Other four
deployments are owned by existing local jobs. Six known pretraining files from
the allowed deployments are excluded. No full temporal/duplicate audit exists;
do not call this an untouched accuracy evaluation.

The support narw is NORTH ATLANTIC, not a verified North Pacific right-whale
example. The sei annotation is Possibly_Detected. Bloop is the NOAA PMEL
icequake recording. Rankings cannot establish species or discovery.

KAGGLE: choose a private notebook, Internet ON, GPU T4 x2. Unzip this bundle to
/kaggle/working/bering_cloud. The notebook should use subprocess with check=True
and unbuffered output, not launch a hidden background process.
Install missing requirements first (especially google-crc32c); the public NOAA
objects expose CRC32C, which is verified. The two pilot generations and byte
lengths are pinned in pilot_source_head.json; total pilot ingress is ~684MB.

1. python /kaggle/working/bering_cloud/scanner.py --verify-only
2. python /kaggle/working/bering_cloud/parity_check.py --output /kaggle/working/cpu_parity.json
3. python /kaggle/working/bering_cloud/scanner.py --max-files 2 --wall-seconds 14400 --output /kaggle/working/bering_results

Do not scale the max-files value until the two-file results and cross-device
precision audit are reviewed. No elapsed-time promise follows from this pilot.
The hard parent limit is four hours. Workers own disjoint explicit keys and
launch one file process at a time on their assigned GPU. Parent Linux process
groups include grandchildren so deadline termination stops inference too.
Default per-file cap is 1800s; download deadline600s, two retries, request
timeout30s. Byte cap4GB/file. Failed items are recorded and never mislabeled
as completed. A failed input remains eligible for a later explicit retry.

The original conversion, spectrogram and matching functions are copied or
extracted verbatim and source-hashed. Sei/narw use the same K=1 windows and
frozen query tokens; Bloop uses its original K=8 resampling separately.
All three methods (v3, raw patch pixels, boxed normalized cross-correlation)
are saved as float32. No float16 output rounding, weight fitting or threshold
changes. TF32 and autocast are disabled. Cross-device equality is NOT assumed.

Each object has three score NPZs and a commit.json binding source SHA256,
GCS generation/MD5 when available, cached1k SHA256, assets and output SHA256.
Raw/resampled temporary FLACs are deleted only after the successful commit
and its hash check, or after a failure is recorded. A resumed output is skipped
only after the asset/object/output checks pass. Do not edit the bundle in place;
its hashes intentionally make modified versions fail verification.

Telemetry includes original download bytes, original/cached audio duration,
conversion, spectrogram and model wall time, processed windows and errors.
Cached1k bytes are not original network bytes. start_s preserves the original
column-offset convention; absolute recording timestamps and STFT-center offsets
need separate checking before interpreting candidate occurrence times.

Keep run_summary.json, per-object commit.json and scores, logs, overlap audit,
protocol, manifest and asset hashes. Stop the Kaggle session after saving outputs.
No paid cloud launch is part of this bundle. The $40 figure is a planning cap.
