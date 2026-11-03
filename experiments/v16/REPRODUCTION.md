# Reconstruction and integrity

## Scope of this draft

`source_manifest.json` binds exact source copies and the fixed synthetic preview. `artifact_manifest.json` separately binds three exact project checkpoints, completed synthetic and real aggregate results/audits, a synthetic figure and an aggregate-only cohort summary. Original and public hashes differ for explicitly documented JSON projections and path normalization. No private absolute paths or real-data rows are included. `publication/verify_source_package.py` checks the staged source copies without accessing the original workspace.

The complete optimizer ledgers and per-query synthetic score arrays are not bundled in this compact package. Published audit summaries bind their original hashes, but replaying those full audits requires the original local artifacts or a separately regenerated run. Do not interpret a hash commitment as inclusion of the committed file.

The generic `scripts/one_reference.py` is independent of Gravity Spy. A compatible project checkpoint and two user-supplied images are sufficient to obtain whole-image similarity. Benchmark cutoffs must not be assumed valid on unrelated images.

## Local benchmark working layout

The unchanged evaluator and data builder expect a working directory named `hypernet_v16` beside a historical `hypernet_v15` directory. Copy this draft into a separate private working tree with that arrangement; the public package directory itself is named `v16`, so its neighbor `v15` alone is not the expected working path.

```text
private_work/
  hypernet_v16/
    hypernet/
    scripts/
    experiments/
    data/gravity_fresh_v16/       # locally rebuilt, never public
    training/                    # locally trained or released project weights
  hypernet_v15/
    data/gravity_spy_v15_final/   # original plan, QC, references, CALIB crops
    data/gravity_spy_official/   # original v1.0/v1.1 metadata
    experiments/gravity_calibration_final/
    checkpoints/details_run01/
```

The v15 source package and its published code are useful starting points, but these private historical inputs are not all bundled here. The v16 evaluator verifies the old calibration score table, sources and checkpoint hashes, plus the enrolled reference pixels. Do not invent missing admission files or replace checksum guards with unverified values. A newly reconstructed protocol must record its own local paths and hashes; a sanitized historical hash record is not executable authorization.

## Official metadata and local consensus

1. Read [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md), obtain the official volunteer CSV and all eight event-metadata CSVs directly from their records, and retain record JSON, publisher checksums and local SHA-256 values. The planner has `volunteer-metadata` and `event-metadata` stages. Its optional combined human/ML HDF5 stage was not used to derive this cohort and is unnecessary for this direct-ID route.
2. Retain the original workflow 2360 version 92.12 snapshot. A live service response can change; a different workflow version cannot silently replace the verified vocabulary. `verified_version_counts` reads this local snapshot. Its creation is a separately documented acquisition step, not an automatic hidden replacement in the planner.
3. Run `ingest-votes` and `verified-version-counts` locally. The compressed annotation CSV is streamed to a private SQLite database. Repeated ballots from the same user do not increase that user's vote; conflicting or multi-label user/event pairs are excluded within the fixed workflow/version. Anonymous votes are excluded. Require at least five consistent registered users and 80% agreement.
4. Restore the historical v15 plan/QC and v1.0 metadata needed for exposure exclusion. Run `candidate-inventory`, which joins official event identity/time/URL metadata, excludes prior-connected groups, and reserves two format-probe groups. The original capped inventory is preserved separately from the final all-eligible amendment.
5. Run `freeze-plan` only when the reproduced metadata inventory matches the audited fixed counts. The final protocol retains all 2,419 eligible distinct time-group representatives and 14 targets, rather than filling an arbitrary budget or relaxing the voting rule. Four-second grouping does not prove physical independence. If upstream files or historical inputs differ, stop and document the discrepancy; do not call a different roster an exact reproduction.
6. To repeat the independent metadata audit, copy the byte-identical `publication/source_snapshots/audit_metadata_independently.py` into `data/gravity_fresh_v16/` and run it there before TEST image acquisition. This script is kept at its original relative working location because it independently resolves the private data directory. It uses Python user sets and independently reconstructed times rather than production SQL or planner functions.

No event-level roster or derived consensus table is in this package. A third party cannot reconstruct the historical cohort from this draft alone. The sources and these dependency details are included so that an appropriately authorized local reconstruction can be audited rather than hidden behind a claimed one-command benchmark.

## Freeze, acquisition and evaluation

Train the primary and equal-budget ablation from scratch, preserve their random state and full synthetic ledgers, and complete the declared synthetic audits. Reuse the fixed old references and 1,579 labeled calibration events to fit each method's cutoffs and select the new-model recipe. Preserve all three candidate recipe traces. These calibration images are additional real supervision, though they never supply a weight gradient.

`scripts/evaluate_v16.py --stage calibration` consumes a locally created model-path JSON and the private fresh plan. `--stage freeze` validates the completed calibration and binds behavior/checkpoints/data. The resulting protocol must also carry the data-builder admission fields: `fresh_plan_sha256`, `model_selection_complete: true`, `test_scores_computed_before_freeze: false`, and actual `behavior_files` / `checkpoint_files` bindings. The local orchestration completes these before acquisition; this draft does not manufacture them.

The paired analysis also requires two supplements created **before the TEST score archive exists**. After calibration, run `analyze_comparison.py --bind-calibration --calibration CALIB_SCORES.npz --test FUTURE_TEST_SCORES.npz --output CALIB_BINDING.json`. After the protocol is frozen, run `analyze_comparison.py --bind-protocol --protocol PROTOCOL.json --binding CALIB_BINDING.json --output PROTOCOL_BINDING.json`. Both commands use the declared comparison plan. Keep these files unchanged; they cannot be recreated after viewing TEST to claim prospective analysis. The final analysis receives both with `--binding` and `--protocol-binding`, along with the same calibration and TEST paths.

After the model and evaluation protocol are frozen, `prepare_fresh_gravity.py fetch-test --protocol PATH --expected-protocol-sha SHA` downloads only the predeclared events with verified HTTPS. Raw 800 by 600 plots are cropped at `[105,66,671,532]`, without resizing. SHA-256 of contiguous uint8 RGB bytes defines pixel identity. The following `blind-quality` stage excludes exact constants, prior/probe duplicates, and all members of within-TEST nonconstant duplicate groups; no replacements or subjective curation are allowed.

Scoring requires an additional local admission binding the unchanged protocol, entire acquired sidecar inventory, blind QC, exact effective event order and `test_scoring_authorized: true`. Then `evaluate_v16.py --stage test` saves all scores locally, and `analyze_comparison.py` reads those local scores under its predeclared paired time-group analysis. Per-event labels, scores and associated plots remain outside the public package. Release only reviewed aggregates and hash commitments after independent audit.

These guards separate original scientific evidence from a new replication. They should not be disabled to make a command appear more convenient. The completed original comparison recovered fewer targets than raw-pixel correlation while making fewer false alarms; it does not establish an overall improvement over that control. A new run requires its own audit and conclusions.
