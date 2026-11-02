"""Local visual report for category-presence experiments; never fits model settings."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import html
import json
import math
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import load_rgb


def _number(value, percent=True):
    if value is None:
        return "—"
    return f"{100*float(value):.1f}%" if percent else f"{float(value):.3f}"


def _font(size):
    for candidate in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def _category_names(manifest):
    names = {}
    categories = manifest.get("categories", [])
    if isinstance(categories, dict):
        categories = [{"category_id": key, "name": value} for key, value in categories.items()]
    for record in categories:
        if isinstance(record, str):
            names[record] = record
        elif isinstance(record, dict):
            cid = str(record.get("category_id", record.get("id", "")))
            value = record.get("display_name", record.get("name", record.get("category_name", cid)))
            names[cid] = str(value)
    return names


def _truth_boxes(row):
    """Read evaluation-only annotations, never draw target boxes on negatives."""
    if not row.get("present"):
        return []
    annotation = row.get("evaluation_annotations") or {}
    values = annotation.get("query_target_boxes", row.get("target_bboxes", []))
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("bbox_xyxy", value.get("xyxy"))
        if isinstance(value, (list, tuple)) and len(value) == 4:
            box = np.asarray(value, dtype=float)
            if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                result.append(box.tolist())
    return result


def _draw_box(draw, box, transform, color, width):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return
    x0, y0, sx, sy, iw, ih = transform
    left, top, right, bottom = box
    left, right = max(0., min(iw, left)), max(0., min(iw, right))
    top, bottom = max(0., min(ih, top)), max(0., min(ih, bottom))
    if right <= left or bottom <= top:
        return
    draw.rectangle((round(x0+left*sx), round(y0+top*sy),
                    round(x0+right*sx), round(y0+bottom*sy)),
                   outline=color, width=width)


def _preview(row, root, path, category_name, outcome):
    reference = Image.fromarray(load_rgb(root/row["reference"]))
    query = Image.fromarray(load_rgb(root/row["query"]))
    canvas = Image.new("RGB", (1080, 600), "#edf2f3")
    draw = ImageDraw.Draw(canvas)
    title_font, font, small = _font(22), _font(16), _font(14)
    draw.text((18, 13), f"Target type: {category_name}"[:88], fill="#15323e", font=title_font)
    draw.text((18, 46), "ONE SUPPORT INPUT", fill="#405d68", font=small)
    draw.text((365, 46), "INDEPENDENT QUERY IMAGE", fill="#405d68", font=small)

    def paste_fit(image, rect):
        x, y, w, h = rect
        original = image.size
        scale = min(w/image.width, h/image.height)
        image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
        ox, oy = x+(w-image.width)//2, y+(h-image.height)//2
        draw.rounded_rectangle((x, y, x+w, y+h), radius=8, fill="white")
        canvas.paste(image, (ox, oy))
        return ox, oy, image.width/original[0], image.height/original[1], original[0], original[1]

    paste_fit(reference, (18, 72, 320, 408))
    transform = paste_fit(query, (365, 72, 697, 408))
    prediction = row.get("prediction") or {}
    proposal = prediction.get("bbox_xyxy")
    if proposal:
        _draw_box(draw, proposal, transform, "#ed8500", 4)
    truths = _truth_boxes(row)
    for box in truths:
        _draw_box(draw, box, transform, "#009b63", 2)
    text = f"{outcome} | {row['model']} | score {row['score']:.3f}"
    draw.text((18, 496), text[:110], fill="#17343f", font=font)
    note = "Orange: model proposal / footprint. Green: target annotation for evaluation only."
    draw.text((18, 526), note, fill="#435e69", font=small)
    label = "Target category present" if row["present"] else "Target category absent"
    decision = "accepted" if row["accepted"] else "rejected"
    draw.text((18, 552), f"{label}; model {decision}. Regions shown are not an accuracy guarantee.", fill="#435e69", font=small)
    canvas.save(path, quality=90)
    return {"has_proposal": bool(proposal), "ground_truth_boxes": len(truths)}


def _select_examples(rows, names, per_outcome=3):
    """Highest-score positives/false positives and lowest-score misses, diversified."""
    selected = []
    for model in names:
        test = [r for r in rows if r["model"] == model and r["split"] == "test"]
        groups = [
            ("recognized", "Recognized target", sorted([r for r in test if r["present"] and r["accepted"]], key=lambda r: (-r["score"], r["id"]))),
            ("missed", "Missed target", sorted([r for r in test if r["present"] and not r["accepted"]], key=lambda r: (r["score"], r["id"]))),
            ("false", "False acceptance", sorted([r for r in test if not r["present"] and r["accepted"]], key=lambda r: (-r["score"], r["id"]))),
        ]
        for kind, label, choices in groups:
            chosen, categories, signatures = [], set(), set()
            # First cover different target categories, then fill remaining slots.
            for diverse in (True, False):
                for row in choices:
                    signature = (row["reference"], row["query"])
                    if len(chosen) >= per_outcome:
                        break
                    if signature in signatures or (diverse and row["category_id"] in categories):
                        continue
                    chosen.append(row);categories.add(row["category_id"]);signatures.add(signature)
            selected.extend((kind, label, r) for r in chosen)
    return selected


def _provenance_value(provenance, key):
    """Read a budget from its declared owner, including CNN wrapper components."""
    value = provenance.get(key)
    if value is not None:
        return value
    for component in ("representation", "cnn"):
        child = provenance.get(component)
        if isinstance(child, dict):
            value = _provenance_value(child, key)
            if value is not None:
                return value
    return None


def _prior_count(provenance):
    value = _provenance_value(provenance, "external_training_images")
    return "Unknown / unverified" if value is None else f"{int(value):,}"


def _prior_source(provenance):
    value = _provenance_value(provenance, "external_pretraining")
    if value is False or (value is None and _provenance_value(provenance, "external_training_images") == 0):
        return "No external visual training"
    if value is True:
        return "External pretraining; see checkpoint provenance"
    return value or "Unknown / unverified; see checkpoint provenance"


def _target_updates(provenance):
    value = _provenance_value(provenance, "target_weight_updates")
    if value is None:
        value = _provenance_value(provenance, "steps_per_reference")
    if value is None and provenance.get("learned_parameters") is False:
        value = 0
    return "See provenance" if value is None else value


def _learning_description(provenance):
    if provenance.get("analytic_correspondences_unchanged") and "raw_score_perturbation_bound" in provenance:
        return "Separate fixed-analytic and learned-CNN scores; analytic matches preserved"
    if provenance.get("analytic_information_retained") or provenance.get("analytic_branch_preserved"):
        weight = provenance.get("effective_learned_weight", provenance.get("learned_weight"))
        if weight == 0:
            return "Exact fixed analytic descriptors; learned branch disabled"
        return "Exact fixed analytic branch retained alongside a support-trained CNN branch"
    if provenance.get("is_cnn") is False and provenance.get("handcrafted_cues_not_learned"):
        return "Fixed analytic image measurements; no CNN or learned feature weights"
    if isinstance(provenance.get("cnn"), dict) and provenance.get("normalization_fit"):
        return "Learned CNN descriptors + frozen support-only normalization"
    if provenance.get("handcrafted_cues_not_learned"):
        if isinstance(provenance.get("cnn"), dict):
            if provenance.get("cnn_weight") == 0:
                return "Fixed shape/texture score; CNN contribution disabled"
            if provenance.get("shape_texture_weight") == 0:
                return "Learned CNN descriptors with fixed region search; shape/texture score disabled"
            return "Hybrid: learned CNN descriptors + fixed shape/texture formulas"
        return "Fixed shape/texture formulas; no learned feature weights"
    if provenance.get("learned_parameters") is False:
        return "Fixed feature rules; no learned feature weights"
    updates = _target_updates(provenance)
    if isinstance(updates, (int, float)) and updates > 0:
        return "Feature weights fitted to the single support image"
    prior = _provenance_value(provenance, "external_training_images")
    if prior is not None and prior > 0:
        return "Previously learned visual representation; see component provenance"
    return "See component provenance"


def _training_notes(models):
    """Describe the actual run rather than inserting a specific model's budget."""
    provenances = [model["provenance"] for model in models.values()]
    counts = [_provenance_value(p, "external_training_images") for p in provenances]
    notes = []
    if counts and all(value == 0 for value in counts):
        notes.append("Every method in this run reports zero external visual training images. "
                     "The table distinguishes support-trained CNN weights from fixed feature formulas.")
    if any(value is not None and value > 0 for value in counts):
        notes.append("Methods with prior visual training have additional image budgets shown above. "
                     "One support image per episode is therefore not one image of total training for those methods. "
                     "Their pretrained categories may overlap this evaluation; development/test separation within this pilot "
                     "does not establish categories unseen during pretraining.")
    if any(value is None for value in counts):
        notes.append("Unknown / unverified prior image counts are not zero-data claims. "
                     "Their external training budget must be established from checkpoint provenance.")
    if any(p.get("handcrafted_cues_not_learned") for p in provenances):
        notes.append("Shape/texture formulas encode hand-designed visual assumptions and do not learn feature weights. "
                     "A hybrid combines these cues with a learned CNN. Its gain must be compared with both standalone components; "
                     "hybrid accuracy alone does not establish how much the CNN learned.")
    if any(p.get("is_cnn") is False and p.get("handcrafted_cues_not_learned") for p in provenances):
        notes.append("The analytic patch control uses fixed image measurements directly, without a CNN. "
                     "Comparing it with the CNN using the same patch extraction and matcher tests the added value of learned descriptors.")
    if any(p.get("analytic_information_retained") or p.get("analytic_branch_preserved") for p in provenances):
        notes.append("Protected variants retain the exact analytic descriptor in separate coordinates alongside the CNN output. "
                     "The learned mixing weight bounds changes in unit-descriptor inner products only. "
                     "Reciprocal match choices and final detection scores can change discontinuously; "
                     "preserving a feature branch does not guarantee unchanged matches or improved category accuracy.")
    if any(p.get("analytic_correspondences_unchanged") and "raw_score_perturbation_bound" in p for p in provenances):
        notes.append("The score guard combines separately completed analytic and learned-only CNN scores. "
                     "Its displayed correspondences belong to the analytic component and do not explain the CNN component. "
                     "The raw-score change is bounded by its learned mixing weight when both components are in [0,1]; "
                     "threshold decisions, false alarms and recall remain unprotected.")
    if any(isinstance(p.get("cnn"), dict) and p.get("normalization_fit") for p in provenances):
        notes.append("Centered CNN variants fit descriptor normalization only on the supplied support. "
                     "The normalization is then frozen for queries; its statistics are an additional support-derived fit, "
                     "not external pretraining or query-based adaptation.")
    notes.append("Compression, slices, warps and other transformations of a support image are derived views, "
                 "not additional observed photographs. Development images and labels used to choose methods or cutoffs "
                 "are separate experimental resources. Equal training-update counts do not guarantee equal computation.")
    return notes


def make_category_report(summary, rows, manifest, output_dir):
    """Write report.html and actual-image JPG comparisons, without metric changes.

    Ground-truth query boxes appear only in the reporting overlay. The model
    predictions and saved decisions are read unchanged from the completed run.
    """
    if not summary.get("completed"):
        raise ValueError("A completed category run is required for a final report")
    output, root = Path(output_dir), Path(manifest["root"])
    output.mkdir(parents=True, exist_ok=True)
    examples = output/"examples"
    examples.mkdir(exist_ok=True)
    esc = lambda value: html.escape(str(value), quote=True)
    names = _category_names(manifest)
    rows = [dict(row) for row in rows]
    for row in rows:
        if "accepted" not in row:
            row["accepted"] = row["score"] >= summary["models"][row["model"]]["calibration"]["threshold"]
    models = summary["models"]
    overall_rows, macro_rows, category_rows, sensitivity_rows, budget_rows, audits = [], [], [], [], [], []
    sensitivity_controls = []
    for alias, model in models.items():
        test, cal, provenance = model["test"], model["calibration"], model["provenance"]
        overall, macro = test["overall"], test["category_macro"]
        overall_rows.append(
            f"<tr><th><code>{esc(alias)}</code></th><td>{overall['tp']}/{overall['n_positive']} · {_number(overall['tpr'])}</td>"
            f"<td>{overall['fp']}/{overall['n_negative']} · {_number(overall['fpr'])}</td>"
            f"<td>{_number(overall['balanced_accuracy'])}</td><td>{_number(overall['auroc'], False)}</td>"
            f"<td>{_number(overall['average_precision'], False)}</td></tr>")
        intervals = test.get("category_cluster_ci", {}).get("intervals", {})
        def ci(key):
            value = intervals.get(key)
            return "—" if not value else f"{_number(value['low'])}–{_number(value['high'])}"
        macro_rows.append(
            f"<tr><th><code>{esc(alias)}</code></th><td>{_number(macro['tpr'])}</td><td>{ci('tpr')}</td>"
            f"<td>{_number(macro['fpr'])}</td><td>{ci('fpr')}</td><td>{_number(macro['balanced_accuracy'])}</td></tr>")
        for category, values in test["by_category"].items():
            category_rows.append(
                f"<tr><td><code>{esc(alias)}</code></td><th>{esc(names.get(category,category))}</th>"
                f"<td>{values['tp']}/{values['n_positive']} · {_number(values['tpr'])}</td>"
                f"<td>{values['fp']}/{values['n_negative']} · {_number(values['fpr'])}</td>"
                f"<td>{_number(values['balanced_accuracy'])}</td></tr>")
        for category, values in test["support_sensitivity"]["per_category"].items():
            same_queries = values.get("identical_query_sets") is True
            sensitivity_controls.append(same_queries)
            def spread(key):
                v = values.get(key)
                return "—" if not v else f"{_number(v['mean'])} ({_number(v['min'])}–{_number(v['max'])})"
            sensitivity_rows.append(
                f"<tr><td><code>{esc(alias)}</code></td><th>{esc(names.get(category,category))}</th>"
                f"<td>{values['n_support_episodes']}</td><td>{'Verified fixed' if same_queries else 'Not isolated'}</td><td>{spread('tpr')}</td><td>{spread('fpr')}</td></tr>")
        prior = _prior_source(provenance)
        updates = _target_updates(provenance)
        budget_rows.append(
            f"<tr><th><code>{esc(alias)}</code></th><td>{_prior_count(provenance)}</td>"
            f"<td>{esc(prior)}</td><td>1 per episode</td><td>{esc(updates)}</td>"
            f"<td>{esc(_learning_description(provenance))}</td></tr>")
        detail = {"calibration": cal, "category_cluster_ci": test.get("category_cluster_ci"),
                  "provenance": provenance, "latency_ms": model.get("latency_ms"),
                  "adaptation_seconds": model.get("adaptation_seconds")}
        audits.append(f"<details><summary>{esc(alias)} — calibration, uncertainty and complete model provenance</summary>"
                      f"<pre>{esc(json.dumps(detail, indent=2))}</pre></details>")
    gallery = []
    selection = _select_examples(rows, models, 3)
    selection_index = []
    for i, (kind, label, row) in enumerate(selection):
        filename = f"category_{i:03d}.jpg"
        category = names.get(str(row["category_id"]), str(row["category_id"]))
        overlay = _preview(row, root, examples/filename, category, label)
        selection_index.append({"file": "examples/"+filename, "pair_id": row["id"], "model": row["model"],
                                "category_id": row["category_id"], "outcome": kind, **overlay})
        box_note = (" Green boxes are evaluation annotations." if overlay["ground_truth_boxes"] else "")
        gallery.append(
            f"<figure class='example' data-model='{esc(row['model'])}' data-outcome='{kind}'>"
            f"<a href='examples/{filename}' target='_blank'><img src='examples/{filename}' "
            f"alt='{esc(label)} for {esc(category)} using {esc(row['model'])}' loading='lazy'></a>"
            f"<figcaption><b>{esc(label)} · {esc(category)}</b><br><code>{esc(row['model'])}</code> · "
            f"episode {esc(row['episode_id'])}<br><span>One support input and a separately sourced query image."
            f"{box_note}</span></figcaption></figure>")
    (output/"example_index.json").write_text(json.dumps(selection_index, indent=2), encoding="utf-8")
    budget = summary.get("budget", {})
    annotation_budget = summary.get("annotation_budget", manifest.get("annotation_budget", {}))
    support_preprocessing = summary.get("support_preprocessing", manifest.get("support_preprocessing", "As supplied by the manifest"))
    support_boxes = annotation_budget.get("support_boxes_per_episode", 0) if isinstance(annotation_budget, dict) else 0
    support_caption = (
        f"This run uses one photograph and {support_boxes} support bounding box{'es' if support_boxes != 1 else ''} per episode. "
        "The annotation defines the support crop and is an additional supervision resource. The query remains the full photograph."
        if support_boxes else
        "The support is the actual reference input supplied by the dataset. Any support preprocessing resources are listed below.")
    test_rows = [r for r in rows if r["split"] == "test"]
    pair_count = len({r["id"] for r in test_rows})
    test_cat_count = len({r["category_id"] for r in test_rows})
    test_episode_count = len({r["episode_id"] for r in test_rows})
    notes = []
    for field in ("interpretation", "dataset_notes", "dataset_limitations"):
        value = summary.get(field, [])
        notes.extend([value] if isinstance(value, str) else value)
    note_html = "".join(f"<li>{esc(note)}</li>" for note in notes)
    training_note_html = "".join(f"<p>{esc(note)}</p>" for note in _training_notes(models))
    model_options = "".join(f"<option value='{esc(name)}'>{esc(name)}</option>" for name in models)
    is_caltech = "caltech" in str(summary.get("manifest", root)).casefold()
    query_medium = "image" if is_caltech else "photograph"
    media_note = ("<p><b>Secondary object-image diagnostic:</b> the Caltech collection includes some illustrated or rendered images. "
                  "It is not exclusively a photograph benchmark, and its typically prominent subjects do not represent arbitrary cluttered scenes.</p>"
                  if is_caltech else "")
    if "coco_scratch_v05" in str(summary.get("manifest", root)).casefold():
        media_note += ("<p><b>Depicted objects:</b> these COCO category labels can include objects shown in artwork. "
                       "An illustrated airplane miss is a museum photograph containing a painted aircraft. "
                       "This cohort is not a verified physical-object-only benchmark; the frozen labels and row inclusion are unchanged.</p>")
    if sensitivity_controls and all(sensitivity_controls):
        sensitivity_note = ("Every reported category uses verified identical query images and labels across its support episodes. "
                            "The observed spread therefore isolates the supplied reference and its reference-dependent adaptation, "
                            "conditional on these fixed query sets, cutoffs and model settings.")
    elif any(sensitivity_controls):
        sensitivity_note = ("Rows marked Verified fixed use identical query images and labels, so their spread isolates reference choice "
                            "and reference-dependent adaptation conditional on those queries. For rows marked Not isolated, query sets differ, "
                            "are unspecified, or only one support episode exists; their spread does not isolate support-photo effects.")
    else:
        sensitivity_note = ("Query sets differ, are unspecified, or only one support episode exists. "
                            "This spread does not isolate support-photo effects.")
    def table(headers, body):
        return "<div class='scroll'><table><thead><tr>"+"".join(f"<th>{esc(h)}</th>" for h in headers)+"</tr></thead><tbody>"+"".join(body)+"</tbody></table></div>"
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HyperNet — one-shot category results</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f5;color:#19333f;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1260px;margin:auto;padding:40px 25px}}h1{{font-size:clamp(34px,5vw,56px);line-height:1.12;max-width:1000px}}
h2{{margin-top:38px;font-size:27px}}a{{color:#006a60}}.lead{{font-size:21px;max-width:1000px}}.eyebrow{{font-size:12px;letter-spacing:.12em}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card,.note,details{{background:white;border:1px solid #d0dfe1;border-radius:10px;padding:20px;margin:18px 0}}
.card strong{{display:block;font-size:25px}}.card span,.muted{{color:#526e79}}.note{{border-left:5px solid #087d6e}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{text-align:left;padding:11px;border-bottom:1px solid #d7e2e5}}
td{{font-variant-numeric:tabular-nums}}code{{font-size:.88em}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}
summary{{cursor:pointer;font-weight:600}}.filters{{position:sticky;top:0;z-index:1;display:flex;flex-wrap:wrap;gap:14px;padding:16px;background:#dfece9;border:1px solid #bdcfcc;border-radius:10px}}
select{{font:inherit;padding:8px;border:1px solid #94afad;border-radius:6px;background:white}}label{{display:flex;align-items:center;gap:8px}}.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin-top:20px}}
figure{{margin:0;background:white;border:1px solid #cbdde1;border-radius:10px;overflow:hidden}}figure img{{width:100%;display:block}}figcaption{{padding:14px;font-size:14px}}figcaption span{{color:#536c77;font-size:13px}}
.example[hidden]{{display:none}}.legend{{display:flex;gap:18px;flex-wrap:wrap}}.orange{{color:#9f5500}}.green{{color:#00794d}}
@media(max-width:760px){{main{{padding:24px 15px}}.cards,.gallery{{grid-template-columns:1fr}}.filters{{position:static}}}}
</style></head><body><main>
<div class="eyebrow">HYPERNET / CATEGORY PRESENCE / {esc(summary.get('created_utc','')[:10])}</div>
<h1>One subject example.<br>A different {query_medium}.</h1>
<p class="lead">Given one support image, does a new full {query_medium} contain any example of the same category? A different instance of the same type counts as a positive.</p>
<p><a href="#examples">See the image comparisons</a> · <a href="summary.json">Complete metrics and data budgets</a> · <a href="predictions.csv">Every prediction</a> · <a href="predictions.jsonl">Detailed evidence</a></p>
<div class="cards"><div class="card"><strong>{test_cat_count} test categories</strong><span>Separate from categories used to choose cutoffs.</span></div>
<div class="card"><strong>{pair_count} test pairs per method</strong><span>{test_episode_count} predefined one-support episodes, with no best-reference selection.</span></div>
<div class="card"><strong>{len(models)} methods</strong><span>All scored on the same category-presence labels.</span></div></div>
<div class="note"><b>What the model receives</b><p>{esc(support_caption)} Query category labels and green annotation boxes are used only to evaluate and illustrate the result; they are never passed to prediction.</p>
<p>Separate source files and exact-content hash checks establish separate recorded inputs. They do not prove different physical individuals or capture sessions, or rule out near-duplicate photographs. This run is a custom category-presence pilot, not the official COCO detection protocol or a detection mAP result.</p>{media_note}</div>
<h2>Held-out category results</h2>
<p>One global cutoff per method is fitted on development categories at a nominal {_number(summary.get('target_dev_fpr'))} false-positive target. The test false-positive rate below is measured, not guaranteed. Target acceptance is recall; false acceptance means the target category is absent but the method accepts it.</p>
{table(['Method','Targets accepted ↑','Absent targets accepted ↓','Balanced accuracy ↑','AUROC','Binary pair AP'], overall_rows)}
<h2>Equal weight for every category</h2>
<p>Category-macro metrics average each category equally. The intervals resample target categories while holding cutoffs fixed. Small category counts and shared query images limit what these intervals establish.</p>
{table(['Method','Macro target acceptance','95% interval','Macro false acceptance','95% interval','Macro balanced accuracy'], macro_rows)}
<h2 id="examples">Inspect the actual images</h2>
<p>Up to three recognized positives, three missed positives and three false acceptances per method, diversified across target categories. Recognized examples and false acceptances start with the highest scores; misses start with the lowest scores. This is a diagnostic selection, not a representative accuracy sample. Click an image to enlarge it.</p>
<p class="legend"><span class="orange">Orange: model crop proposal or projected footprint</span><span class="green">Green: target-category boxes for evaluation only</span></p>
<p class="muted">An orange region is not a verified object box. Methods without a region prediction have no orange overlay. No target-category box is drawn on a negative example.</p>
<div class="filters"><label>Method <select id="modelFilter"><option value="all">All methods</option>{model_options}</select></label>
<label>Outcome <select id="outcomeFilter"><option value="all">All outcomes</option><option value="recognized">Recognized target</option><option value="missed">Missed target</option><option value="false">False acceptance</option></select></label><span id="visibleCount" aria-live="polite"></span></div>
<div class="gallery">{''.join(gallery)}</div><p id="emptyGallery" hidden>No selected examples match these filters.</p>
<details><summary>Per-category results</summary>{table(['Method','Target category','Targets accepted','Absent targets accepted','Balanced accuracy'], category_rows)}</details>
<details><summary>Sensitivity to which single support image is supplied</summary><p>Mean (minimum–maximum) across predefined one-support episodes. Each prediction still receives one support image; the references are not combined. {esc(sensitivity_note)}</p>{table(['Method','Target category','Support episodes','Query-set control','Target acceptance mean (range)','False acceptance mean (range)'], sensitivity_rows)}</details>
<h2>Training and annotation budgets</h2>
{table(['Method','Prior visual training images','Prior training source','Target support images','Target training updates','Learned and fixed components'], budget_rows)}
{training_note_html}
<details><summary>Support preprocessing and annotation resources</summary><pre>{esc(json.dumps({'support_preprocessing':support_preprocessing,'annotation_budget':annotation_budget,'episode_budget':budget},indent=2))}</pre><p>Adapter-level flags describe which arguments the model receives. They do not erase annotations used upstream to produce the support crop.</p></details>
<h2>Interpretation and audit trail</h2><ul>{note_html}</ul>{''.join(audits)}
<details><summary>Dataset validation and environment</summary><pre>{esc(json.dumps({'manifest_sha256':summary.get('manifest_sha256'),'category_splits':summary.get('category_splits'),'validation':summary.get('validation'),'versions':summary.get('versions'),'seed':summary.get('seed')},indent=2))}</pre></details>
<p class="muted">These are measured results from an engineering pilot. Model names, category names and example labels come from the saved run. No accuracy is inferred from example appearance or from a cited model's published benchmark.</p>
</main><script>
const modelSelect=document.getElementById('modelFilter');
const outcomeSelect=document.getElementById('outcomeFilter');
function filterExamples(){{
 let visible=0;
 document.querySelectorAll('.example').forEach(card=>{{
  const keep=(modelSelect.value==='all'||card.dataset.model===modelSelect.value)&&(outcomeSelect.value==='all'||card.dataset.outcome===outcomeSelect.value);
  card.hidden=!keep;if(keep)visible++;
 }});
 document.getElementById('visibleCount').textContent=visible+' examples';
 document.getElementById('emptyGallery').hidden=visible!==0;
}}
modelSelect.addEventListener('change',filterExamples);outcomeSelect.addEventListener('change',filterExamples);filterExamples();
</script></body></html>"""
    (output/"report.html").write_text(content, encoding="utf-8")
    return output/"report.html"

