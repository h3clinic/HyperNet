# One-shot whale call search (OnceNet working name)

Search years of unlabeled ocean recordings using **one** example call. The model (`whale_ssl_v3`) is trained label-free on spectrograms (self-supervised). It never sees a label for the target call. You draw one box around one call, and a block-cosine box matcher ranks every 4 s window in the archive.

**Status: candidates, not confirmed detections.** No whale expert has reviewed them yet.

## Main result so far: sei-like downsweeps in the Bering Sea, autumn 2004

The reference is one labeled sei whale downsweep from DCLDE 2013: `NOPP6_EST_20090402_194500_CH10.wav` at 140–141 s. We searched the NOAA/Scripps (MBARC) passive acoustic recordings from the southeastern Bering Sea:

| Mooring and deployment | Period | Searched |
|---|---|---|
| M2, Bering 2 | Oct 2004 – Jan 2005 | done |
| M4, Bering 2 | Oct – Dec 2004 | done |
| M2, Bering 3 | Apr – Sep 2005 | done |
| M2, Bering 4 | from late Sep 2005 | running |

- **20 sei-like candidate calls.** They sweep from about 60 Hz down to 25–35 Hz in about 1 s.
  - 14 on 25 Oct 2004 at M2, in 4 bouts.
  - 1 on 29 Oct 2004 at M4.
  - 1 on 18 Nov 2004 at M2.
  - 4 on 21 Nov 2004 at M2.
- **Fin whale control.** June 2005 (216 h at M2) gives similar match scores, but those calls end at 44–50 Hz, compared with 19–35 Hz for the candidates. The search score alone does not separate the species; the shape test in `code/fair_measure.py` does. See `june_control_results.json`, `fair_measure.json` and `june_control_hits.png`.
- **2005 (Apr–Sep):** almost all strong matches are fin-like. One borderline call on 15 Jul 2005 ends at 38 Hz (`july2005_candidate.png`).
- **Cross-check with Google's multispecies whale model** (trained, 12 classes, no sei class), run on the found clips only: inconclusive. Its fin class fired on 0 of 20 candidates, 0 of 17 fin-like calls, 0 of 84 background windows and 0 of 3 labeled sei calls. See `google_validation.json` and `pretrained/validate_google.py`.
- **Prior reports:** we found no published sei whale (or 65→35 Hz downsweep) detections from these moorings for 2004–06.

Limits: the recordings were cached at 1 kHz. Species is inferred from call shape only. The shape cutoffs (end ≤38 Hz and width ≥18 Hz for sei-like; end ≥42 Hz for fin-like) were set by us, not by an expert.

## What's here

| Path | Contents |
|---|---|
| `code/` | spectrograms, one-shot matcher (`oneshot.py`), archive scan (`scan_oneshot.py`, `scan_loop.py`), Bering download (`cache_bering.py`), controls and shape test, video maker |
| `whale_ssl_v1..v3/` | self-supervised training code, logs, protocol and the selected checkpoint (intermediate checkpoints omitted) |
| `pretrained/` | baselines with existing trained models; Google model cross-check |
| `examples/` | the one-shot reference windows and boxes (sei, right whale upcall, Bloop) |
| `*.json`, `*.log`, `*.sha256` | results, logs and frozen hashes |
| `*.png` | contact sheets of top matches and controls |
| `listen/` | short audio clips (4× faster and real speed) |
| `video/bering_sei_candidates.mp4` | 97 s summary video |
| `bering/`, `dclde2013/` | inventories, file metadata and label index (no audio) |

Not included because of size (about 47 GB): the downloaded audio, the 1 kHz cache, per-file score files and the training corpus. The audio is public: MBARC Bering recordings come from the NOAA passive acoustic archive, and DCLDE 2013 comes from its official release. `code/cache_bering.py` re-downloads the Bering files.

Scripts expect to be run from `code/`, with the data folders next to it, as in the original workspace.
