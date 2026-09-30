"""Label-free CN7 production-date split; reuse transfer_risk estimators unchanged."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import transfer_risk as tr


def split_dates(frame):
    frame = frame.copy()
    dates = pd.to_datetime(frame['TimeStamp'], errors='raise')
    if dates.isna().any():
        raise ValueError('Missing TimeStamp')
    frame['production_date'] = dates.dt.strftime('%Y-%m-%d')
    days = sorted(frame.production_date.unique())
    if len(days) < 12:
        raise ValueError(f'Need at least 12 production dates; found {len(days)}')
    early, late = days[:6], days[-6:]
    assert not set(early) & set(late)
    return frame, days, early, late


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--public', required=True)
    ap.add_argument('--out', default='outputs/analysis/cn7_time_split')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-perm', type=int, default=500)
    args = ap.parse_args()
    cfg = json.loads((tr.ROOT / 'config.json').read_text(encoding='utf-8'))
    features = [f for f in tr.competition_features(cfg) if f not in tr.NON_FEATURES]
    assert len(features) == 23
    # Explicit allowlist: no quality labels or defect proxies are loaded.
    df = tr._public.read_csv(Path(args.public), 'unlabeled_data.csv',
                            usecols=lambda c: c in ['TimeStamp', 'PART_NAME'] + features)
    if set(features) - set(df.columns):
        raise ValueError('Missing process features')
    df = df[df.PART_NAME.astype(str).str.upper().str.extract(r'(CN7|RG3)', expand=False).eq('CN7')]
    df, days, early, late = split_dates(df)
    if not np.isfinite(df[features].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite process features')
    cross_dates = int(df.groupby(features, dropna=False).production_date.nunique().gt(1).sum())
    if cross_dates:
        raise ValueError('Identical process vectors span dates; date assignment needs review')
    shots = tr.dedupe_shots(df, features)
    daily = df.groupby('production_date').size().rename('n_rows').to_frame()
    daily['n_shots'] = shots.groupby('production_date').size()
    daily['group'] = ['cn7_early' if d in early else 'cn7_late' if d in late else 'excluded' for d in daily.index]
    A = tr.dedupe_shots(df[df.production_date.isin(early)], features)
    B = tr.dedupe_shots(df[df.production_date.isin(late)], features)
    sd = pd.concat([A[features], B[features]]).std(ddof=0)
    constant = sd[sd.eq(0)].index.tolist()
    feats = [f for f in features if f not in constant]
    sc = StandardScaler().fit(pd.concat([A[feats], B[feats]]))
    XA, XB = sc.transform(A[feats]), sc.transform(B[feats])
    print('Selected shots:', len(A), len(B), flush=True)
    w = tr.per_variable_w1(A, B, feats)
    print('MMD permutations', flush=True)
    m = tr.mmd_rbf(XA, XB, n_perm=args.n_perm, seed=args.seed)
    print('Domain classifiers', flush=True)
    auc, coef = tr.domain_auc(XA, XB, feats, seed=args.seed)
    print('Within-period random-half references', flush=True)
    baselines = {name: tr.within_domain_baseline(part, feats, args.seed, min(args.n_perm, 200))
                 for name, part in [('cn7_early', A), ('cn7_late', B)]}
    result = dict(source=str(Path(args.public).resolve()), member='dataset/unlabeled_data.csv',
                  n_cn7_rows=len(df), n_production_dates=len(days), all_dates=days,
                  early_dates=early, late_dates=late, excluded_dates=days[6:-6],
                  n_shots_early=len(A), n_shots_late=len(B), features=feats,
                  original_features=features, constant_features=constant,
                  cross_date_duplicate_vectors=cross_dates, label_used=False,
                  dedupe='drop_duplicates on original 23 process features within group',
                  seed=args.seed, n_perm=args.n_perm, baseline_n_perm=min(args.n_perm, 200),
                  mmd_max_rows_per_group=2000, w1_std_mean=float(w.w1_std.mean()),
                  w1_std_max=float(w.w1_std.max()), **m, **auc, baselines=baselines,
                  transfer_risk_sha256=hashlib.sha256(Path(tr.__file__).read_bytes()).hexdigest(),
                  limitations=['Shot-weighted, not equal date weights',
                               'Random shot CV and permutations do not account for temporal dependence',
                               'Cross-mold reference uses separately standardized competition data; populations differ'])
    out = tr.ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(f'{out}_daily_counts.csv')
    w.merge(coef, on='feature').to_csv(f'{out}_per_variable.csv', index=False)
    pd.DataFrame(baselines).T.to_csv(f'{out}_no_shift.csv', index_label='group')
    pd.DataFrame([{k: v for k, v in result.items() if not isinstance(v, (dict, list))}]).to_csv(f'{out}_summary.csv', index=False)
    Path(f'{out}_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
