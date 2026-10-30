"""Step 14.27: frozen-policy cross-camera validation on a later WILDTRACK clip.

No production changes. All alternatives are fixed *before* looking at this clip.
Oracle boxes and within-camera IDs isolate global association error from detection.

    python -m src.identity_validation prepare --download
    python -m src.identity_validation extract --threads 4 --affinity 0,2,4,6,8,10
    python -m src.identity_validation run
    python -m src.identity_validation report
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

from .association_ablation import Policy, baseline_replay, replay, score_subset
from .evaluate import HF_BASE
from .identity_evaluation import CAMERAS, DATA, DEFAULT_OUT, dump_json, ground_truth, source_hash
from .reid import MATCH_MARGIN, MATCH_THRESHOLD, MODEL_WEIGHTS
from .reid_diagnostics import (DEFAULT_DIAG, extract as extract_features,
                                read_features, sha256)
from .source_consistency import SourceGuard, replay_source

# Predeclared late frame set. Previous experiments used 0,5,...,495.
# WILDTRACK annotations are provided for IDs 0,5,...,1995.
LATE_IDS = tuple(range(1400, 2000, 10))
FIRST_100_ORACLE = DEFAULT_OUT / 'oracle_100'
PRIOR_STEP_25 = DEFAULT_OUT / 'source_consistency_100'
OUT = DEFAULT_OUT / 'identity_validation_late60'
ORACLE = OUT / 'oracle'
FEATURES = OUT / 'features'
POLICY = Policy('camera_balanced_batch_hungarian_t80_m05', threshold=.80,
                margin=.05, prototype='camera_balanced', assignment='batch_hungarian')
GUARD = SourceGuard('source_2of3_t70', 'two_of_three', cosine_floor=.70,
                    history_count=3, min_agree=2)
POLICIES = ('production_ema_t85_m05', 'experimental_balanced_hungarian_t80_m05',
            'experimental_source_2of3_t70')


def fixed_frame_ids(pilot=False):
    return list(LATE_IDS[:15] if pilot else LATE_IDS)


def assert_later_frames(ids, old_ids):
    if not ids or ids != sorted(set(ids)):
        raise ValueError('Late frame IDs must be sorted and unique')
    if not set(ids).issubset(set(LATE_IDS)):
        raise ValueError('Validation frame IDs differ from predeclared late segment')
    if set(ids) & set(old_ids):
        raise ValueError('Evaluation overlaps the 0-495 development frame set')
    if old_ids and max(old_ids) >= min(ids):
        raise ValueError('Validation is not temporally separated from development')


def previous_gt(data_dir, prior_oracle_dir):
    path = Path(prior_oracle_dir) / 'oracle_manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Prior oracle manifest missing: {path}')
    old = json.loads(path.read_text(encoding='utf-8'))
    frame_ids = [int(fid) for fid in old['frame_ids']]
    if len(frame_ids) != 100 or frame_ids != list(range(0, 500, 5)):
        raise ValueError('Expected original 100-frame baseline (0,5,...,495)')
    gt = ground_truth(data_dir, frame_ids)
    if source_hash(gt) != old['gt_sha256']:
        raise ValueError('Original 100-frame annotation checksum changed')
    return frame_ids, gt, old['gt_sha256']


def identities(gt):
    return {int(p['gt_id']) for cams in gt.values() for camera in CAMERAS
            for p in cams[camera]}


def get_assets(data_dir, ids):
    root = Path(data_dir)
    paths = []
    for fid in ids:
        stem = f'{fid:08d}'
        paths.append((f'annotations_positions/{stem}.json',
                      root / 'annotations_positions' / f'{stem}.json'))
        paths.extend((f'Image_subsets/{cam}/{stem}.png',
                      root / 'Image_subsets' / cam / f'{stem}.png') for cam in CAMERAS)
    return paths


def fetch_assets(data_dir, ids, timeout=90):
    """Download only the predeclared later subset (4 PNGs + GT per frame)."""
    files = get_assets(data_dir, ids)
    for i, (remote, dest) in enumerate(files, 1):
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + '.part')
        print(f'[{i}/{len(files)}] Download {remote}', flush=True)
        try:
            request = urllib.request.Request(f'{HF_BASE}/{remote}',
                headers={'User-Agent': 'multicam-identity-validation/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open('wb') as fp:
                while chunk := response.read(1024*1024):
                    fp.write(chunk)
            if temporary.stat().st_size == 0:
                raise ValueError(f'Empty download: {remote}')
            temporary.replace(dest)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def validate_assets(data_dir, ids):
    missing = [str(dest) for _, dest in get_assets(data_dir, ids)
               if not dest.is_file() or dest.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f'{len(missing)} late-clip files missing; run prepare --download. First: {missing[0]}')


def prepare(data_dir=DATA, old_oracle_dir=FIRST_100_ORACLE, output_dir=OUT,
            download=False, pilot=False):
    ids = fixed_frame_ids(pilot)
    previous_ids, previous, prior_sha = previous_gt(data_dir, old_oracle_dir)
    assert_later_frames(ids, previous_ids)
    if download:
        fetch_assets(data_dir, ids)
    validate_assets(data_dir, ids)
    late = ground_truth(data_dir, ids)
    prior_ids, seen = identities(previous), identities(late)
    new_ids = seen - prior_ids
    out = Path(output_dir)
    (out / 'oracle').mkdir(parents=True, exist_ok=True)
    manifest = {
        'purpose': 'Predeclared later WILDTRACK temporal clip; not independent camera/site',
        'mode': 'pilot' if pilot else 'full',
        'frame_ids': ids, 'frame_count': len(ids),
        'previous_frame_ids_sha256': prior_sha,
        'gt_sha256': source_hash(late),
        'old_unique_gt_ids': len(prior_ids),
        'late_unique_gt_ids': len(seen),
        'novel_gt_ids': sorted(new_ids),
        'novel_gt_id_count': len(new_ids),
        'shared_gt_id_count': len(seen & prior_ids),
        'policy_names': list(POLICIES),
        'limitations': [
            'No WILDTRACK frame set from the first 100 annotations is reused.',
            'Same WILDTRACK site, cameras and potentially some of the same people.',
            'Novel-person metric labels exclude identities present in the old 100 frames; all persons remain in the matcher gallery.',
            'Oracle annotated boxes and perfect within-camera tracks are used, not live YOLO/ByteTrack detections.',
            'A 15-frame pilot is only a smoke test, not a final evaluation.',
        ]
    }
    dump_json(out / 'oracle' / 'oracle_manifest.json', {'frame_ids': ids,
        'gt_sha256': manifest['gt_sha256'], 'selection': manifest['mode']})
    dump_json(out / 'validation_manifest.json', manifest)
    print(f'Prepared {len(ids)} late frame sets: {len(new_ids)} novel identities, '
          f'{len(seen & prior_ids)} previously seen identities', flush=True)
    return manifest


def check_manifest(data_dir, prior_oracle_dir, output_dir):
    out = Path(output_dir)
    file = out / 'validation_manifest.json'
    if not file.is_file():
        raise FileNotFoundError(f'Run prepare before extraction: {file}')
    m = json.loads(file.read_text(encoding='utf-8'))
    previous_ids, gt_old, sha_old = previous_gt(data_dir, prior_oracle_dir)
    if m['previous_frame_ids_sha256'] != sha_old:
        raise ValueError('Prior annotation SHA changed since prepare')
    ids = list(map(int, m['frame_ids']))
    assert_later_frames(ids, previous_ids)
    if ids != fixed_frame_ids(pilot=(m['mode'] == 'pilot')):
        raise ValueError('Validation sample deviates from fixed pilot/full plan')
    if source_hash(ground_truth(data_dir, ids)) != m['gt_sha256']:
        raise ValueError('Late annotations changed since prepare')
    old_ids, late_ids = identities(gt_old), identities(ground_truth(data_dir, ids))
    if sorted(late_ids - old_ids) != m['novel_gt_ids']:
        raise ValueError('Identity split differs from prepare')
    return m


def extract(data_dir=DATA, old_oracle_dir=FIRST_100_ORACLE,
            output_dir=OUT, threads=4, affinity='0,2,4,6,8,10'):
    m = check_manifest(data_dir, old_oracle_dir, output_dir)
    return extract_features(data_dir, Path(output_dir) / 'oracle',
                            Path(output_dir) / 'features', threads, affinity)


def prior_policy_check(prior_dir, expected_policy=POLICY, expected_guard=GUARD):
    path = Path(prior_dir) / 'comparison.json'
    if not path.is_file():
        raise FileNotFoundError(f'Step 14.25 reference not found: {path}')
    report = json.loads(path.read_text(encoding='utf-8'))
    if not report.get('baseline_parity'):
        raise ValueError('Prior matcher replay parity not verified')
    from dataclasses import asdict
    if report.get('base_policy') != asdict(expected_policy):
        raise ValueError('The experimental policy differs from frozen Step 14.25 reference')
    if not any(x.get('guard') == asdict(expected_guard)
               for x in report.get('experiments', [])):
        raise ValueError('The 0.70 source guard was not verified in Step 14.25')
    return report


def summarize_score(p, labels, gt, ids):
    return score_subset(p, labels, gt, set(ids))


def summary_row(name, segment, scores, decisions=None):
    keys = ['true_pairs', 'predicted_pairs', 'tp', 'fp', 'fn', 'precision',
            'recall', 'f1', 'unique_correctly_linked_gt_identities',
            'unique_wrongly_merged_identity_pairs']
    return {'policy': name, 'subset': segment, **{k: scores[k] for k in keys},
            'source_vetoes': (decisions or {}).get('source_veto', 0)}


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run(data_dir=DATA, old_oracle_dir=FIRST_100_ORACLE, output_dir=OUT,
        prior_dir=PRIOR_STEP_25, old_features_dir=DEFAULT_DIAG):
    out = Path(output_dir)
    m = check_manifest(data_dir, old_oracle_dir, out)
    old_report = prior_policy_check(prior_dir)
    feat_manifest, gt, by_frame = read_features(out / 'features', data_dir, out / 'oracle')
    if feat_manifest['ground_truth_sha256'] != m['gt_sha256']:
        raise ValueError('Late features do not match prepared annotations')
    if feat_manifest['frame_ids'] != m['frame_ids']:
        raise ValueError('Partial feature extraction: expected all validation frames')
    if not Path(MODEL_WEIGHTS).is_file():
        raise FileNotFoundError(f'Missing frozen production checkpoint {MODEL_WEIGHTS}')
    current_checkpoint_sha = sha256(MODEL_WEIGHTS)
    if feat_manifest['checkpoint_sha256'] != current_checkpoint_sha:
        raise ValueError('Checkpoint differs from late-clip extraction')
    original_features_manifest = json.loads(
        (Path(old_features_dir) / 'embeddings_manifest.json').read_text(encoding='utf-8'))
    if current_checkpoint_sha != original_features_manifest['checkpoint_sha256']:
        raise ValueError('Late validation does not use same OSNet x0.25 weights as Step 14.20')
    if old_report['source']['checkpoint_sha256'] != current_checkpoint_sha:
        raise ValueError('Step 14.25 policy was audited with a different OSNet checkpoint')
    _, gt_old, _ = previous_gt(data_dir, old_oracle_dir)
    novel = identities(gt) - identities(gt_old)
    all_ids = identities(gt)

    # Frozen policies: no thresholds or parameters may be chosen from this clip.
    baseline_preds, baseline_labels = baseline_replay(by_frame)
    experimental, experiment_preds, experiment_labels = replay_source(by_frame, POLICY, None)
    # Fail closed if the source-consistency wrapper ever changes unguarded replay.
    reference_preds, reference_labels, _  = replay(by_frame, POLICY)
    if experimental.attempts is None or experiment_preds != reference_preds or experiment_labels != reference_labels:
        raise RuntimeError('Experimental policy parity failed on unseen clip')
    guarded, guard_preds, guard_labels = replay_source(by_frame, POLICY, GUARD)
    if baseline_labels != experiment_labels or baseline_labels != guard_labels:
        raise ValueError('GT label order changed between replay implementations')
    preds = [baseline_preds, experiment_preds, guard_preds]
    results = {}
    table = []
    by_camera = []
    for name, prediction, extra in zip(POLICIES, preds, ({}, dict(experimental.decisions), dict(guarded.decisions))):
        scores = {'all_identities': summarize_score(prediction, baseline_labels, gt, all_ids),
                  'novel_identities': summarize_score(prediction, baseline_labels, gt, novel)}
        results[name] = {'scores': scores, 'decisions': extra}
        for segment, score in scores.items():
            table.append(summary_row(name, segment, score, extra))
            for pair, metrics in score['camera_pairs'].items():
                by_camera.append({'policy': name, 'subset': segment, 'camera_pair': pair,
                                  **{k: metrics.get(k, 0) for k in ('true_pairs', 'tp', 'fp')}})
        print(f'{name}: all TP/FP={scores["all_identities"]["tp"]}/{scores["all_identities"]["fp"]}; '
              f'novel TP/FP={scores["novel_identities"]["tp"]}/{scores["novel_identities"]["fp"]}', flush=True)
    report = {
        'purpose': 'Step14.27 locked-policy validation on later WILDTRACK frames; offline oracle tracking',
        'mode': m['mode'],
        'frozen_config_source': str(Path(prior_dir) / 'comparison.json'),
        'evaluation_source': {'frame_ids': m['frame_ids'], 'frame_count': len(m['frame_ids']),
                              'gt_sha256': m['gt_sha256'],
                              'features_sha256': feat_manifest['features_sha256'],
                              'checkpoint_sha256': current_checkpoint_sha,
                              'novel_gt_id_count': len(novel),
                              'shared_gt_id_count': len(all_ids & identities(gt_old)),
                              'novel_persons_present_in_gt': sorted(novel)},
        'reference_parity': True,
        'configs': {'production': {'threshold': MATCH_THRESHOLD, 'margin': MATCH_MARGIN,
                                  'prototype': 'EMA', 'assignment': 'greedy',
                                  'min_observations': 3, 'refresh_every': 5},
                    'experimental': POLICY.__dict__, 'guard': GUARD.__dict__},
        'results': results,
        'decision_rule': {
            'minimum_descriptive_pair_precision': 0.9,
            'requires_nonzero_predictions': True,
            'no_automatic_deployment': True,
            'notes': 'Do not pick parameters using this validation segment; new dataset/site and real detections remain required.'},
        'limitations': m['limitations'] + [
            'Pair observations across frames are correlated; precision and recall do not imply statistical certainty.',
            'Only four of seven WILDTRACK camera views used.',
            'No detection failures, ByteTrack ID switches, RTSP latency, FPS or MQTT delivery included.',
            'Not official identity F1/HOTA; only per-frame cross-camera pair precision/recall.',
            'If no novel true pairs exist, novel-person recall is undefined; do not claim independent identity validation.',
        ]
    }
    dump_json(out / 'comparison.json', report)
    write_csv(out / 'comparison.csv', table,
        ['policy', 'subset', 'true_pairs', 'predicted_pairs', 'tp', 'fp', 'fn',
         'precision', 'recall', 'f1', 'unique_correctly_linked_gt_identities',
         'unique_wrongly_merged_identity_pairs', 'source_vetoes'])
    write_csv(out / 'camera_pair_metrics.csv', by_camera,
        ['policy', 'subset', 'camera_pair', 'true_pairs', 'tp', 'fp'])
    return report


def report(output_dir=OUT):
    obj = json.loads((Path(output_dir) / 'comparison.json').read_text(encoding='utf-8'))
    print(f'Step14.27 mode={obj["mode"]}, frames={obj["evaluation_source"]["frame_count"]}, '
          f'novel persons={obj["evaluation_source"]["novel_gt_id_count"]}')
    for name, result in obj['results'].items():
        for subset, s in result['scores'].items():
            p = 'undefined' if s['precision'] is None else f'{s["precision"]:.4f}'
            print(f'{name} | {subset}: TP={s["tp"]} FP={s["fp"]} '
                  f'precision={p} recall={s["recall"]} '
                  f'wrong-identity-pairs={s["unique_wrongly_merged_identity_pairs"]}')
    if obj['mode'] == 'pilot':
        print('PILOT ONLY: results not a final validation.')
    print('Production has not been modified.')
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'extract', 'run', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--data-dir', type=Path, default=DATA)
        p.add_argument('--output-dir', type=Path, default=OUT)
        if name != 'report':
            p.add_argument('--old-oracle-dir', type=Path, default=FIRST_100_ORACLE)
        if name == 'prepare':
            p.add_argument('--download', action='store_true',
                           help='Fetch missing 60 late frame sets (images may total hundreds of MB)')
            p.add_argument('--pilot', action='store_true',
                           help='ONLY 15 late frame sets; for smoke tests, not final results')
        if name == 'extract':
            p.add_argument('--threads', type=int, default=4)
            p.add_argument('--affinity', default='0,2,4,6,8,10')
        if name == 'run':
            p.add_argument('--prior-dir', type=Path, default=PRIOR_STEP_25)
            p.add_argument('--old-features-dir', type=Path, default=DEFAULT_DIAG)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        prepare(args.data_dir, args.old_oracle_dir, args.output_dir,
                download=args.download, pilot=args.pilot)
    elif args.command == 'extract':
        extract(args.data_dir, args.old_oracle_dir, args.output_dir, args.threads, args.affinity)
    elif args.command == 'run':
        run(args.data_dir, args.old_oracle_dir, args.output_dir,
            args.prior_dir, args.old_features_dir)
    else:
        report(args.output_dir)


if __name__ == '__main__':
    main()
