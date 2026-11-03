# Gravity Spy one-reference discovery run over O3b

**Verdict under the frozen rule: not confirmed** (1 of 3 primary types passed; the rule required 2).

What the run did find: for **1400 Ripples**, given five labeled examples from the 2018 training set, the label-free hybrid search placed **2 volunteer-majority 1400 Ripples glitches that neither Gravity Spy CNN had labeled as 1400 Ripples** (the CNN called them 1080Lines and Whistle, at confidence 0.55 and 0.62) into its top 200 of 38,291 eligible O3b events. Chance expectation was 0.06 (hypergeometric p = 0.0014); the CNN's own P(1400 Ripples) ranking found 0. For Paired Doves it found 1 such glitch at rank 3 (expected 0.07, p = 0.067, comparator 0), labeled Fast_Scattering by the CNN. For Helix it found none. Across all five types, 415 of the 1,000 top-200 candidates have never been classified by any human; they are delivered in `candidates.csv` and count as zero finds here.

## Question and pass rule (frozen before the search)

Does a one-reference search over the uncertain part of O3b concentrate volunteer-supported rare glitches that neither Gravity Spy CNN labeled as that type?

> POSITIVE only if, in at least 2 of the 3 primary types (Paired_Doves, 1400Ripples, Helix), the hypergeometric tail probability of the hybrid top-200 new_majority count (within the evaluated subset, on that ranking's own covered events) is below 0.05/3 AND the hybrid new_majority@200 counted per 4 s group exceeds the CNN P(type) comparator's on the same subset and K. Otherwise "not confirmed". Chirp is descriptive until an injection cross-match exists; Wandering_Line has too few volunteer-majority events (4 pool-wide).

`protocol.json` (SHA-256 `72e10ffb…`) was written after the pool, arrays, both scripts, the verifier's constants, the truth tiers, the evaluated subset and this rule were fixed, and before any pool event was scored. An adversarial review of the protocol before the freeze found and fixed two blockers: the arms were originally credited with v0.4's four-duration benchmark numbers although they score one view, and "new find" was judged against the wrong CNN.

## Pool

38,471 O3b events (of 292,261) whose Gravity Spy CNN confidence is below 0.7 or whose CNN probability for a rare type is at least 0.3 while labeled otherwise; 32,375 distinct 4 s groups; one duration (0.5 s). The rule was hashed into `pool.json` before any image was fetched. All 38,471 images were fetched (0 failures) and converted; none was blank. 18,043 pool events (47%) have at least one volunteer classification; the median is 2 volunteers.

## Arms

Three label-free arms from v0.4/v0.5, scored on the single 0.5 s view with the mean of per-reference z-scores over the five v1.1 references of each type. Feature vectors are built exactly as in v0.4's Bank. A single 0.5 s view with five-reference averaging was never a v0.4 candidate, so the arms were first measured on the v0.5 O3b TEST set (a disclosed second read of that set): hybrid 0.599, random 0.581, trained 0.547 mean AP (the v0.5 one-reference fused4 numbers were 0.500 / 0.495 / 0.477). Hybrid is the pre-declared primary arm.

## Truth

Volunteer-only, from Zenodo 5911227 (MD5 checked): each volunteer's pick is the argmax of their confusion-weighted row in `tracks`; the CNN prior (row 0) is excluded; events with no volunteer row are uncovered. Primary tier **majority** = at least 2 volunteers and a strict majority. Two different CNNs label the pool (the O3b tables' newer model with Fast_Scattering and Blip_Low_Frequency, and the 22-class model volunteers worked under; they agree on 41% of covered events), so **new** requires both labels to differ from the type. The vocabulary-gap share (O3b label Fast_Scattering or Blip_Low_Frequency, classes volunteers had no button for) is reported: 4 of the 5 Paired Doves "supported" finds are such events; the one majority find is too.

Evaluation is within the subset the O3b CNN did not label as the type, for the arms and the CNN P(type) comparator alike. Each ranking's own covered count sets a hypergeometric null. Counts are given per event and per 4 s group.

Disclosed: per-event volunteer picks for the five types over the whole pool were extracted and seen before the protocol was written (`pre_freeze_pool_volunteer_picks.csv`, hash in `protocol.json`); the type list, K and tiers were chosen knowing the pool ceilings. The search itself read no labels.

## Results

Primary cells (hybrid, K = 200, majority tier, eligible subset):

| Type | New majority finds (groups) | Covered of 200 | Expected by chance | p | CNN P(type) comparator | Pool ceiling (majority) | Pass |
|---|---|---|---|---|---|---|---|
| Paired_Doves | 1 (1) | 107 | 0.069 | 0.067 | 0 | 11 | no |
| **1400Ripples** | **2 (2)** | 92 | 0.056 | **0.0014** | 0 | 11 | **yes** |
| Helix | 0 | 119 | 0.106 | 1.0 | 0 | 16 | no |

Secondary grid (descriptive), new finds per tier majority / confirmed / supported within the eligible subset:

| Type | K | hybrid | random | trained | CNN P(type) | pool ceiling |
|---|---|---|---|---|---|---|
| Paired_Doves | 20 | 1/1/1 | 0/0/0 | 0/0/0 | 0/0/0 | 11/14/147 |
| Paired_Doves | 200 | 1/1/5 | 1/2/6 | 1/1/1 | 0/0/0 | 11/14/147 |
| 1400Ripples | 20 | 0/0/0 | 0/0/0 | 0/0/2 | 0/0/0 | 11/17/83 |
| 1400Ripples | 200 | 2/2/4 | 1/1/2 | 1/2/12 | 0/0/1 | 11/17/83 |
| Chirp | 200 | 0/0/0 | 0/0/0 | 0/1/1 | 1/2/2 | 14/26/92 |
| Helix | 200 | 0/0/1 | 0/0/2 | 0/0/4 | 0/0/0 | 16/21/257 |
| Wandering_Line | 200 | 0/0/1 | 2/2/3 | 0/0/3 | 0/0/1 | 3/4/24 |

Under the weaker "supported" tier (one volunteer can suffice), the hybrid top-200 is also above chance for Paired_Doves (5 vs 0.9 expected, p = 0.002) and 1400Ripples (4 vs 0.4, p = 0.001); the trained arm's 12 supported 1400Ripples finds is the largest single count. These tiers were declared descriptive before the run and are not claims.

The three majority finds:

| Type | Rank | Event | Detector | O3b CNN label (confidence) | CNN P(type) | Volunteers (picks of type) |
|---|---|---|---|---|---|---|
| Paired_Doves | 3 | hk10B1TbKg | L1 | Fast_Scattering (0.55) | 0.00003 | 3 (2) |
| 1400Ripples | 115 | m3flaaBnQo | H1 | 1080Lines (0.55) | 0.00001 | 2 (2) |
| 1400Ripples | 119 | cmOrxUwWzS | L1 | Whistle (0.62) | 0.0045 | 2 (2) |

In all three the CNN's own probability for the type was essentially zero, so these are events the official pipeline would never have surfaced for that class. Each rests on two volunteers' agreement on the same image.

## Candidates

`candidates.csv` holds the hybrid top-200 per type within the eligible subset with CNN label and confidence, P(type), volunteer status and the image URL. `gallery.html` shows the first 20 per type in rank order; it embeds the publisher's images, so it is kept local and not committed. Unclassified candidates per type: 1400Ripples 108, Paired_Doves 93, Helix 81, Wandering_Line 77, Chirp 56. For 1400Ripples and Paired_Doves the rank-1 candidate has never been classified by a human. Classifying these is a separate study and would need its own frozen protocol and arm-blind reviewers.

## Reading this

- The pool holds very few needles under the strict tier (11, 11, 16 volunteer-majority finds among about 17,000–18,000 covered events per type), so even a perfect searcher could show at most a handful in a 200-event list. The 1400Ripples cell is a clear above-chance concentration; the Paired_Doves cell is suggestive and is also a vocabulary-gap case; Helix shows nothing.
- The CNN P(type) comparator finds nothing in any primary cell, because volunteer-supported finds almost all have P(type) below 0.01. The honest claim is "the search surfaces volunteer-supported rare candidates where the CNN's own probability is near zero", not "the search beats the CNN".
- Volunteer votes are two people's picks on the same image; a reviewer should treat the three finds as candidates to confirm, not confirmed glitches. The 415 never-classified candidates are the actual discovery deliverable.
- Chirp results are descriptive pending a cross-match against the public injection segments; Wandering_Line has too few needles to test.
- Paired_Doves reference 5 (the seeded replacement chosen in v0.4 on the 1.0 s view) is nearly blank at the 0.5 s view used here, so that type effectively ran on four references. The two 1400Ripples finds share the faint high-frequency dot visible in all five 1400Ripples references.

## Reproduce

```bash
python scripts/prepare_gravity_spy_discovery.py --stage roster --tables <O3b tables> --output <pool>
python scripts/prepare_gravity_spy_discovery.py --stage fetch  --output <pool>
python scripts/prepare_gravity_spy_discovery.py --stage arrays --output <pool>
python scripts/gravity_spy_discovery.py --stage measure --data <v1.1> --pool <pool> --o3 <v0.5 cohort> --runs <hypernet_v14/checkpoints> --trained <hypernet_gs_v04/checkpoints> --output <out>
python scripts/gravity_spy_discovery.py --stage freeze  ...same...
python scripts/gravity_spy_discovery.py --stage search  ...same...
python scripts/gravity_spy_discovery_verify.py --pool <pool> --volunteer <5911227 hdf5> --output <out>
python scripts/gravity_spy_discovery_gallery.py --pool <pool> --data <v1.1> --output <out>
```

The search took about 30 minutes on CPU. `scores.npz` keeps every event's score for every arm and type; `verification.json` holds every cell with its null; `rankings.json` the frozen top-200 lists over the full pool.
