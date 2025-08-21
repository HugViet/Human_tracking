
import os
import argparse
import motmetrics as mm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-dir', required=True, help='Path to MOT17 split folder, e.g. datasets/MOT17/train')
    ap.add_argument('--res-dir', required=True, help='Directory containing results <SEQ>.txt')
    ap.add_argument('--dist-thresh', type=float, default=0.5, help='IoU threshold for matching (default 0.5)')
    ap.add_argument('--save-csv', default=None, help='Optional path to save CSV summary')
    args = ap.parse_args()

    accs, names = [], []
    seqs = [d for d in sorted(os.listdir(args.gt_dir)) if os.path.isdir(os.path.join(args.gt_dir, d))]
    for seq in seqs:
        gt_file = os.path.join(args.gt_dir, seq, 'gt', 'gt.txt')
        res_file = os.path.join(args.res_dir, f'{seq}.txt')
        if not (os.path.isfile(gt_file) and os.path.isfile(res_file)):
            continue

        gt = mm.io.loadtxt(gt_file, fmt='mot16', min_confidence=1)
        res = mm.io.loadtxt(res_file, fmt='mot16')
        acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=args.dist_thresh)
        accs.append(acc)
        names.append(seq)

    if not accs:
        print('No matching sequences found. Check --gt-dir and --res-dir.')
        return

    mh = mm.metrics.create()
    metrics = [
        'mota', 'motp', 'idf1', 'precision', 'recall',
        'num_switches',
        'mostly_tracked', 'mostly_lost',
        'num_false_positives', 'num_misses', 'num_fragmentations'
    ]
    summary = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=True
    )
    print(mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    ))


    if args.save_csv:
        summary.to_csv(args.save_csv) 
        print(f'Saved CSV summary to: {args.save_csv}')

if __name__ == '__main__':
    main()
