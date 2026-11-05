# Research behind v0.4

A multi-agent literature study run on 2026-11-02/03 before v0.4 was designed: 13 research threads (9 planned, 4 added by a completeness critic), 191 findings, each actionable finding fact-checked against its source by a separate agent (137 supported, 53 rejected or corrected, 1 unchecked).

- `research_report.md`: the synthesized report, written only from verified findings. Sections: what we know, why HyperNet's training hurt, ranked v0.4 design candidates, fresh test data plan, false-alarm plan, evaluation changes, open risks.
- `synthesis.json`: the seven ranked design candidates with mechanism, evidence URLs, data-budget change and cost; the fresh-data and false-alarm plans; the evaluation checklist; comparable published numbers; the critic's follow-up questions.
- `findings_unverified_list.json`: every finding as the researchers returned it, with URL, title, year and self-rated confidence. Verification verdicts are in the workflow journal, not reproduced here.

What v0.4 took from it: candidate 1 (family-member positives with a many-positive objective) and the 2×2 control arm (new data, old objective); candidate 4 (per-arm normalization, five random seeds, fusion bar, blank rule); part of candidate 2 (no flips or rotations in training, four-duration fusion). Candidates 3 (coverage diagnostic for the generator), 6 (declared in-domain unlabeled pretraining) and 7 (physics injections through the Gravity Spy Q-transform) are untested proposals.

Sources from 2025–2026 were read from arXiv HTML and may be newer than the agents' training data; the report flags them.
