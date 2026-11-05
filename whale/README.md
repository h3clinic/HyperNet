# One-shot whale call search (OnceNet working name)

Search years of unlabeled ocean recordings using **one** example call. The model (`whale_ssl_v3`) was pretrained label-free on 77,000 real acoustic windows. Model and configuration selection separately used 10 labeled support examples and 6,641 query windows. Each deployed search enrolls one boxed reference call, and a block-cosine box matcher ranks 4 s archive windows.

**Status: candidates, not confirmed detections.** No whale expert has reviewed them yet.

## Main result so far: sei-like downsweeps in the Bering Sea, autumn 2004

The reference is a DCLDE 2013 downsweep annotated **Possibly_Detected** for sei whale: `NOPP6_EST_20090402_194500_CH10.wav` at 140–141 s. Its source and species remain uncertain. The search used NOAA/Scripps (MBARC) passive acoustic recordings from the southeastern Bering Sea:

| Mooring and deployment | Period | Searched |
|---|---|---|
| M2, Bering 2 | Oct 2004 – Jan 2005 | done |
| M4, Bering 2 | Oct – Dec 2004 | done |
| M2, Bering 3 | Apr – Sep 2005 | done |
| M2, Bering 4 | Sep 2005 – Jan 2006 | done |

All 3,007 recording files (about 13 months of audio) have been searched.

- **20 sei-like candidate calls.** They sweep from about 60 Hz down to 25–35 Hz in about 1 s.
  - 14 on 25 Oct 2004 at M2, in 4 bouts.
  - 1 on 29 Oct 2004 at M4.
  - 1 on 18 Nov 2004 at M2.
  - 4 on 21 Nov 2004 at M2.
- **Fin-like morphology control.** June 2005 (216 h at M2) gives similar match scores, but those calls end at 44–50 Hz, compared with 19–35 Hz for the candidates. The investigator-defined rule in `code/fair_measure.py` separates these shape groups; neither that rule nor the search score establishes species. See `june_control_results.json`, `fair_measure.json` and `june_control_hits.png`.
- **2005 (Apr–Sep):** almost all strong matches are fin-like. One borderline call on 15 Jul 2005 ends at 38 Hz (`july2005_candidate.png`).
- **Sep 2005 – Jan 2006:** of the top 60 matches, 45 are fin-like, 13 unclear (mostly fin 20 Hz pulses) and 2 sei-like (`bering04_top60.json`). The sei-like ones are a clean 70→35 Hz downsweep on 6 Jan 2006 at about 10:07 and a 65→25 Hz sweep among fin calls on 7 Jan 2006 at about 07:11. A sweep on 29 Sep 2005 (55→18 Hz) is also possible. See `final_new_candidates.png` and `autumn2005_check.png`.
- **Overall top 40 across all 4 deployments** (`new_candidates_final.json`): the sei-like matches are all from Oct–Nov 2004. The 2005–06 calls above rank lower.
- **Cross-check with Google's multispecies whale model** (trained, 12 classes, no sei class), run on the found clips only: inconclusive. Its fin class fired on 0 of 20 candidates, 0 of 17 fin-like calls, 0 of 84 background windows and 0 of 3 labeled sei calls. See `google_validation.json` and `pretrained/validate_google.py`.
- **Prior-record check:** the branch review did not locate published sei whale or 65→35 Hz downsweep detections from these moorings for 2004–06. This bounded search does not establish novelty or exclude an earlier report.

Limits: the recordings were cached at 1 kHz. Species is inferred from call shape only. The shape cutoffs (end ≤38 Hz and width ≥18 Hz for sei-like; end ≥42 Hz for fin-like) were set by us, not by an expert.

## What's here

| Path | Contents |
|---|---|
| `code/` | spectrograms, one-shot matcher (`oneshot.py`), archive scan (`scan_oneshot.py`, `scan_loop.py`), Bering download (`cache_bering.py`), controls and shape test, video maker |
| `whale_ssl_v1..v3/` | self-supervised training code, logs and protocols; the selected v3 checkpoint is linked below |
| `pretrained/` | baselines with existing trained models; Google model cross-check |
| `examples/` | the one-shot reference windows and boxes (sei, right whale upcall, Bloop) |
| `*.json`, `*.log`, `*.sha256` | results, logs and frozen hashes |
| `*.png` | contact sheets of top matches and controls |
| `listen/` | short audio clips (4× faster and real speed) |
| `video/bering_sei_candidates.mp4` | 97 s summary video |
| `bering/`, `dclde2013/` | inventories, file metadata and label index (no audio) |

The frozen [selected acoustic checkpoint](../oncenet/current/bering_inference/whale/whale_ssl_v3/selected.pt) is included in the current reproduction package; the original branch does not contain its model weight files.

Not included because of size (about 47 GB): the downloaded audio, the 1 kHz cache, per-file score files and the training corpus. The audio is public: MBARC Bering recordings come from the NOAA passive acoustic archive, and DCLDE 2013 comes from its official release. `code/cache_bering.py` re-downloads the Bering files.

Scripts expect to be run from `code/`, with the data folders next to it, as in the original workspace.
