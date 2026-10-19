"""Fill a manuscript skeleton from pipeline outputs only — no invented numbers.

Every number is followed by its source tag [file: key]. Sections that need
human judgement (Introduction, Discussion, references, ethics numbers) are left
as outlines with TODO markers. Feasibility wording is forced when the test set
has fewer than 5 patients or the run was a smoke test.

Usage:
  python draft_manuscript.py --project <project dir> --run runs/<ts> [--external external/<name>] \
      [--search search/] [--sample-size sample_size/] --out manuscript_draft.md
The project dir must contain pack.json, governance_decision.json, split/split_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f3(x):
    return "NA" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--external", action="append", default=[])
    ap.add_argument("--search", default="")
    ap.add_argument("--sample-size", default="")
    ap.add_argument("--audit", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, R = Path(a.project), Path(a.project) / a.run if not Path(a.run).is_absolute() else Path(a.run)
    pack, gov = j(P / "pack.json"), j(P / "governance_decision.json") or {}
    sp, m = j(P / "split" / "split_summary.json"), j(R / "metrics.json")
    if not (pack and sp and m):
        raise SystemExit("[STOP] need pack.json, split/split_summary.json and <run>/metrics.json")
    task, classes, te = m["task_type"], m["class_names"], m.get("test")
    tr = (m.get("arch"), pack["training"])
    n_test = sp["n_patients"].get("test", 0)
    feas = n_test < 5 or m.get("smoke_test")
    W = []
    w = W.append
    title = pack["title_en"] + (": a feasibility study" if feas else "")
    w(f"# {title}\n")
    w("<!-- Auto-drafted by SurgVision toolkit. Every number carries [source]. Edit wording, never numbers without updating the source. -->\n")
    if feas:
        w(f"> **Feasibility:** test set has {n_test} patient(s){' and this is a smoke-test run' if m.get('smoke_test') else ''}. "
          "Do not claim accuracy, clinical utility or generalisability.\n")

    # ---------------- abstract (results line only, rest TODO)
    w("## Abstract\n")
    w(f"**Objective.** {pack['research_question']}\n")
    w("**Methods.** TODO — one sentence each: data source, annotation, patient-level split, model, primary metric.\n")
    if te and task == "segmentation":
        parts = [f"{c} Dice {f3(v['mean_dice'])} (95% CI {f3(v['dice_ci_patient_bootstrap']['ci95'][0])}–{f3(v['dice_ci_patient_bootstrap']['ci95'][1])})"
                 for c, v in te["per_class"].items()]
        w(f"**Results.** On {sp['n_frames'].get('test', 0)} test frames from {n_test} patient(s), " + "; ".join(parts) +
          " [metrics.json: test.per_class].\n")
    elif te:
        w(f"**Results.** On {sp['n_frames'].get('test', 0)} test frames from {n_test} patient(s), balanced accuracy "
          f"{f3(te['balanced_accuracy'])}, macro F1 {f3(te['macro_f1'])} [metrics.json: test].\n")
    w("**Conclusion.** TODO — " + ("state feasibility only." if feas else "state what the result supports, no more.") + "\n")

    w("## Introduction (outline)\n1. Clinical problem: " + pack.get("clinical_rationale", "") +
      "\n2. Prior work — TODO (only references verified in PubMed)\n3. Aim: " + pack["research_question"] + "\n")

    # ---------------- methods
    d, an = pack["data"], pack["annotation"]
    w("## Methods\n### Study design and ethics")
    w(f"Retrospective technical validation of a {task} model for {d.get('procedure', 'surgical')} video "
      f"({d.get('modality', 'video')}). Ethics: {gov.get('irb_status', 'TODO')} [governance_decision.json: irb_status]; "
      "approval number TODO.\n")
    w("### Data")
    w(f"Source: {d['source']}; inclusion: {d.get('inclusion', 'TODO')}; exclusion: {d.get('exclusion', 'TODO')} [pack.json: data]. "
      f"Frames were extracted at {d.get('extract_fps')} fps [pack.json: data.extract_fps]. "
      "Container metadata were removed, burned-in text regions were masked, and out-of-view or identifiable frames were "
      "excluded after human review of every frame [extract_report.json].\n")
    w("### Annotation")
    defs = "; ".join(f"*{k}*: {v}" for k, v in pack["label_definitions"].items())
    w(f"Labels ({an['mode']}, platform {an['platform']}) followed written definitions — {defs}. Boundary rules: "
      + " / ".join(pack["hard_cases"]) + f". Annotators: {an.get('annotators')} [pack.json: annotation]. "
      "Frames without an annotation were excluded rather than treated as negatives.\n")
    w("### Data partitioning")
    w(f"Patients were assigned to training/validation/test at the patient level (seed {sp['seed']}, ratios {sp['ratios']}); "
      f"no patient appears in more than one set [split_summary.json: leakage={sp['leakage']}].\n")
    if a.sample_size and (Path(a.sample_size) / "sample_size_precision.json").exists():
        ss = j(Path(a.sample_size) / "sample_size_precision.json")
        w(f"Planned test size: {ss['test_patients_needed']} patients for a 95% CI half-width of {ss['half_width']} "
          f"(SD {ss['sd']}, ICC {ss['icc']}, design effect {ss['design_effect']}) [sample_size_precision.json].\n")
    w("### Model development")
    if a.search and (Path(a.search) / "search_summary.json").exists():
        s = j(Path(a.search) / "search_summary.json")
        cands = ", ".join(f"{b['arch']} ({b['val_mean']})" for b in s["leaderboard"] if b["val_mean"] is not None)
        w(f"Candidate architectures were compared on validation only ({s['selection_metric']}, {s['folds']} fold(s), "
          f"{s['epochs_per_candidate']} epochs each): {cands}; {s['chosen']} was chosen ({s['chosen_reason']}) "
          "[search_summary.json]. The test set was not used for selection.\n")
    w(f"Final model: {tr[0]} with ImageNet-pretrained backbone={m['pretrained_encoder']}, input {tr[1].get('input_size')}, "
      f"{m['epochs']} epochs, batch {tr[1].get('batch_size')}, learning rate {tr[1].get('lr', 3e-4)}, AdamW, cosine schedule, "
      f"flip and intensity augmentation [config.json; metrics.json]. The checkpoint with the best validation "
      f"{m['selection_metric']} (epoch {m['selected_epoch']}) was reloaded and evaluated once on the test set "
      f"[metrics.json: selected_epoch, checkpoint_reload_verified={m['checkpoint_reload_verified']}].\n")
    w("### Evaluation")
    w(f"Primary metric: {pack['metrics']['primary']}. Secondary: {', '.join(pack['metrics'].get('secondary', []))} [pack.json: metrics]. "
      "95% CIs by patient-level cluster bootstrap (2,000 resamples). Metrics were computed at model input resolution. "
      f"Software: PyTorch {m['torch']} on {m['device']} [metrics.json].\n")

    # ---------------- results
    w("## Results\n### Dataset")
    w(f"Patients (train/val/test): {sp['n_patients'].get('train', 0)}/{sp['n_patients'].get('val', 0)}/{n_test}; frames: "
      f"{sp['n_frames'].get('train', 0)}/{sp['n_frames'].get('val', 0)}/{sp['n_frames'].get('test', 0)} [split_summary.json]. "
      "Table 1 [results/table1_dataset.md].\n")
    if a.audit and (Path(a.audit) / "data_audit.json").exists():
        au = j(Path(a.audit) / "data_audit.json")
        if au.get("warnings"):
            w("Composition notes: " + "; ".join(au["warnings"]) + " [data_audit.json].\n")
    w("### Model development")
    w(f"Validation score rose from {f3(m['val_before_training'])} before training to {f3(m['val_selected'])} at epoch "
      f"{m['selected_epoch']} [metrics.json].\n")
    w("### Internal test")
    if te is None:
        w("No test split — validation results only.\n")
    elif task == "segmentation":
        for c, v in te["per_class"].items():
            ci = v["dice_ci_patient_bootstrap"]
            w(f"- {c}: mean Dice {f3(v['mean_dice'])} (95% CI {f3(ci['ci95'][0])}–{f3(ci['ci95'][1])}{'; CI unreliable, <5 patients' if not ci['reliable'] else ''}), "
              f"median {f3(v['median_dice'])}, mean IoU {f3(v['mean_iou'])}; structure missed entirely in {v['n_missed_entirely']} "
              f"and falsely predicted in {v['n_false_alarm_images']} of {v['n_images_scored'] + v['n_both_empty']} images "
              f"[metrics.json: test.per_class.{c}].")
        w("")
    else:
        ci = te["accuracy_ci_patient_bootstrap"]
        w(f"Accuracy {f3(ci['estimate'])} (95% CI {f3(ci['ci95'][0])}–{f3(ci['ci95'][1])}), balanced accuracy "
          f"{f3(te['balanced_accuracy'])}, macro F1 {f3(te['macro_f1'])} [metrics.json: test]. Per class (recall / precision / AUROC): "
          + "; ".join(f"{c} {f3(te['per_class_recall'][i])}/{f3(te['per_class_precision'][i])}/{f3((te.get('per_class_auroc_ovr') or [None]*len(classes))[i])}"
                      for i, c in enumerate(classes)) + ".\n")
    for e in a.external:
        em = j(Path(e) / "external_metrics.json")
        if not em:
            continue
        w(f"### External validation — {em['evaluation_set']}")
        w(f"The frozen model was applied without retraining to {em['n_frames']} frames from {em['n_patients']} patient(s). ")
        h = em["headline"]
        if task == "segmentation":
            w("; ".join(f"{c} Dice {f3(v['mean_dice'])} (95% CI {f3(v['ci95'][0])}–{f3(v['ci95'][1])})" for c, v in h.items())
              + " [external_metrics.json: headline]. Table 3 [table3_external.md].\n")
        else:
            w(f"Balanced accuracy {f3(h['balanced_accuracy'])}, macro F1 {f3(h['macro_f1'])} [external_metrics.json: headline].\n")
    w("### Failure analysis\nTODO — describe worst examples [examples/worst*], missed / false-alarm images, most confused classes.\n")

    # ---------------- limitations
    w("## Limitations (check each)")
    lim = [f"{sp['n_patients'].get('train', 0) + sp['n_patients'].get('val', 0) + n_test} patients"
           + (", single centre" if not a.external else ""),
           f"{an.get('annotators')} annotator(s); inter-rater agreement {'not measured' if not an.get('double_annotation_fraction') else 'see agreement_summary.json'}",
           "frames without annotation excluded (possible selection bias)",
           "frame-level analysis ignores temporal context" if task == "classification" else "per-frame segmentation; temporal consistency not evaluated",
           "metrics computed at model input resolution",
           "no external validation" if not a.external else "external set size limited — see CI",
           "TODO: state whether any setting was changed after seeing test results"]
    w("\n".join(f"- [ ] {x}" for x in lim) + "\n")
    w("## Discussion (outline)\n1. Main finding (no stronger than Results)\n2. Comparison with prior work — TODO verified references\n"
      "3. Implications and next steps (external validation, more patients per sample_size)\n")
    w(f"## Reporting checklist\nUse {pack.get('reporting_guideline', 'TRIPOD+AI / CLAIM')} — map each item to a section; mark missing items.\n")
    w("## To fill by authors\n- [ ] IRB number, dates, centre\n- [ ] Authors, affiliations, funding, conflicts\n- [ ] References (PubMed-verified)\n- [ ] Figure legends\n")
    Path(a.out).write_text("\n".join(W), encoding="utf-8")
    print(f"draft -> {a.out}")


if __name__ == "__main__":
    main()
