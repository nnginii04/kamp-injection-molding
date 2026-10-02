"""Export the existing class-distance ambiguity using main's saved block folds.

Run: python -m src.analysis.export_ambiguity
No model training, new scoring method, or persisted split generation.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.ambiguity import FEATURES, score_fold
from src.cv import cv_splits, make_groups
from src.data import assign_shots, sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis'
STEM = 'ambiguity_scores_for_ablation'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def protected_hashes():
    paths = {ROOT / p for p in git('ls-files').decode().splitlines()}
    paths.update(p for p in (ROOT / 'outputs').rglob('*') if p.is_file())
    paths.update(p for p in (ROOT / 'data/raw').glob('*') if p.is_file())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)
            if p.is_file() and not p.name.startswith(STEM)}


def main():
    before = protected_hashes()
    main_commit = git('rev-parse', 'main').decode().strip()
    cfg = json.loads(git('show', f'{main_commit}:config.json'))
    assert (cfg['cv']['n_splits'], cfg['cv']['n_repeats']) == (3, 5)
    # Reproduce main's split ONLY in memory for comparison; scoring below uses
    # the saved fold columns exclusively. Never overwrite or create split files.
    for path in ['src/cv.py', 'src/data.py']:
        assert (ROOT / path).read_bytes().replace(b'\r\n', b'\n') == git('show', f'{main_commit}:{path}').replace(b'\r\n', b'\n'), path
    folds = pd.read_csv(ROOT / 'outputs/fold_id.csv')
    cols = [f'fold_block_r{r}' for r in range(5)]
    assert not folds[['mold', 'row_id', 'shot_id', 'block_id', *cols]].isna().any().any()
    assert not folds.duplicated(['mold', 'row_id']).any()
    baseline = json.loads((ROOT / 'outputs/manifest.json').read_text(encoding='utf-8'))
    saved_pred = pd.read_csv(ROOT / 'outputs/predictions/pipeline_logreg.csv')
    saved_pred = saved_pred[saved_pred.cv_mode.eq('block')]
    output, audits, counts, inputs = [], [], [], {}
    for product, files in cfg['products'].items():
        path = ROOT / cfg['data_dir'] / files['labeled']
        inputs[str(path.relative_to(ROOT))] = sha256(path)
        assert sha256(path) == baseline['input_sha256'][files['labeled']]
        data = pd.read_csv(path).rename(columns={cfg['id_col']: 'row_id', cfg['label_col']: 'y'})
        original = [c for c in data if c not in ['row_id', 'y']]
        assert set(original) - {'Clamp_Open_Position'} == set(FEATURES)
        assert set(data.y.unique()) <= {0, 1}
        data.y = data.y.eq(cfg['positive_label']).astype(int)
        data = assign_shots(data, original, product)
        data['product'] = product
        f = folds[folds.mold.eq(product)]
        data = data.merge(f.drop(columns='mold'), on=['row_id', 'shot_id'], how='outer', validate='one_to_one', indicator=True)
        assert data._merge.eq('both').all()
        data = data.drop(columns='_merge').reset_index(drop=True)
        assert np.isfinite(data[FEATURES].to_numpy(float)).all()
        assert data.groupby('shot_id')[FEATURES].nunique().le(1).all().all()
        np.testing.assert_array_equal(make_groups(data, 'block', cfg['cv']['block_shots']), data.block_id)
        assert set(saved_pred[saved_pred['product'].eq(product)].repeat) == set(range(5))
        for repeat, col in enumerate(cols):
            assert set(data[col]) == {0, 1, 2}
            assert data.groupby('shot_id')[col].nunique().eq(1).all()
            assert data.groupby('block_id')[col].nunique().eq(1).all()
            # main algorithm + main seed + main raw data must exactly reproduce
            # every stored block fold assignment, not just its fold count.
            for fold, (_, va) in enumerate(cv_splits(data, 'block', cfg, repeat)):
                assert set(data.iloc[va].row_id) == set(data.loc[data[col].eq(fold), 'row_id'])
            old = saved_pred[saved_pred['product'].eq(product) & saved_pred.repeat.eq(repeat)]
            m = old.merge(data[['row_id', 'shot_id', 'y', col]], on='row_id', how='outer', validate='one_to_one', suffixes=('', '_saved'), indicator=True)
            assert (m._merge.eq('both') & m.shot_id.eq(m.shot_id_saved) & m.fold.eq(m[col]) & m.y_true.eq(m.y)).all()
            for fold in range(3):
                reference = data.loc[data[col].ne(fold), ['product', 'row_id', 'shot_id', 'y', *FEATURES]].copy()
                valid = data[data[col].eq(fold)]
                # One query per shot; no validation label/side reaches scoring.
                query = valid.drop_duplicates('shot_id')[['product', 'row_id', 'shot_id', *FEATURES]].copy()
                assert set(reference.y) == {0, 1}
                assert set(reference.shot_id).isdisjoint(valid.shot_id)
                assert set(data.loc[data[col].ne(fold), 'block_id']).isdisjoint(valid.block_id)
                scores, params = score_fold(reference, query)
                assert set(params['train_shot_ids']) == set(reference.shot_id)
                expected_mean = reference.drop_duplicates('shot_id')[FEATURES].mean().to_numpy()
                np.testing.assert_allclose(params['mean'], expected_mean, atol=1e-12)
                # Audit invariance to query labels/side and other query samples.
                poisoned = query.assign(y=999, side='DO_NOT_USE', PassOrFail=-999)
                alternate, alternate_params = score_fold(reference, poisoned)
                pd.testing.assert_frame_equal(scores, alternate)
                assert params == alternate_params
                single, single_params = score_fold(reference, query.iloc[:1])
                pd.testing.assert_frame_equal(scores.iloc[:1].reset_index(drop=True), single)
                assert params == single_params
                # Verify original row-query behavior gives the same score for
                # both products without exporting duplicated rows.
                row_scores, _ = score_fold(reference, valid[['product', 'row_id', 'shot_id', *FEATURES]])
                assert row_scores.groupby('shot_id').ambiguity_score.nunique().eq(1).all()
                np.testing.assert_allclose(row_scores.ambiguity_score,
                    row_scores.shot_id.map(scores.set_index('shot_id').ambiguity_score), rtol=0, atol=0)
                for name in ['nearest_normal_shot_id', 'nearest_defect_shot_id']:
                    assert set(scores[name]) <= set(reference.shot_id)
                assert scores.ambiguity_score.between(0, 1).all()
                export = scores[['shot_id', 'ambiguity_score']].rename(columns={'ambiguity_score': 'ambiguity'})
                export['row_id'] = pd.NA  # no arbitrary representative product ID
                export['mold'] = product
                export['repeat'] = repeat
                export['fold'] = fold
                output.append(export[['row_id', 'shot_id', 'mold', 'repeat', 'fold', 'ambiguity']])
                counts.append(dict(mold=product, repeat=repeat, fold=fold, validation_rows=len(valid), validation_shots=len(query), export_records=len(export)))
                audits.append(dict(mold=product, repeat=repeat, fold=fold, train_rows=len(reference),
                    train_shots=len(params['train_shot_ids']), validation_shot_ids=query.shot_id.tolist(),
                    query_label_side_invariance=True, query_batch_invariance=True, row_mapping_equivalent=True, **params))
    result = pd.concat(output, ignore_index=True).sort_values(['mold', 'repeat', 'fold', 'shot_id']).reset_index(drop=True)
    assert not result.duplicated(['mold', 'repeat', 'shot_id']).any()
    assert not result[['shot_id', 'mold', 'repeat', 'fold', 'ambiguity']].isna().any().any()
    assert result.groupby(['mold', 'shot_id']).size().eq(5).all()
    for product, g in result.groupby('mold'):
        for repeat, part in g.groupby('repeat'):
            expected = folds[folds.mold.eq(product)].drop_duplicates('shot_id').set_index('shot_id')[cols[repeat]].sort_index()
            actual = part.set_index('shot_id').fold.sort_index()
            pd.testing.assert_series_equal(actual, expected, check_names=False)
    assert before == protected_hashes(), 'Protected files changed'
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{STEM}.csv'
    result.to_csv(path, index=False)
    pd.DataFrame(counts).to_csv(OUT / f'{STEM}_counts.csv', index=False)
    meta = dict(main_commit=main_commit, branch=git('branch', '--show-current').decode().strip(),
        existing_implementation='src/analysis/ambiguity.py:score_fold (unchanged)',
        existing_runner_split='fold_shot_r*: not block; no previous ambiguity output found in current checkout',
        train_fold_only=True, main_split_exact_match=True, baseline_oof_assignments_exact_match=True,
        fold_scheme='3-fold x 5 repeat block', fold_sha256=sha256(ROOT / 'outputs/fold_id.csv'), input_sha256=inputs,
        features=FEATURES, unit='shot', row_id='intentionally blank; key is mold/repeat/shot_id',
        definition='2*min(d_normal,d_defect)/(d_normal+d_defect); both zero -> 1',
        reference='TRAIN row class membership retained: mixed-label train shots belong to both class sets',
        missing_ambiguity=int(result.ambiguity.isna().sum()), intentional_blank_row_id=len(result), duplicate_keys=0,
        minimum=float(result.ambiguity.min()), maximum=float(result.ambiguity.max()),
        product_counts={p:dict(unique_shots=int(g.shot_id.nunique()),records=len(g),minimum=float(g.ambiguity.min()),maximum=float(g.ambiguity.max())) for p,g in result.groupby('mold')},
        every_validation_shot_exactly_once_per_repeat=True, protected_files_unchanged=True,
        protected_file_count=len(before), protected_sha256=before, output_sha256=sha256(path), fold_audit=audits)
    (OUT / f'{STEM}_verification.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    note = f'''# Ambiguity export for inspection-priority ablation

기존 score_fold/계산식을 변경하지 않고 main {main_commit}의 저장 block fold로 다시 계산했다.
기존 실행기는 fold_shot_r*를 사용한다. 5-repeat를 5-fold로 해석하면 안 된다. 기존 결과 파일은 현재 checkout에서 발견되지 않았다.
main의 config와 cv_splits로 메모리에서 재현한 fold, 저장 fold_id.csv, baseline Logistic block OOF의 배정이 모두 일치한다. split 파일은 생성/수정하지 않았다.

StandardScaler는 해당 validation fold를 제외한 train 고유 Shot만 fit한다. 두 class 최근접 거리도 해당 train Shot만 참조한다.
train의 제품 y는 정상/불량 참조군을 구분하는 데만 사용한다. validation y/side는 query에서 제거했다. 23개 공정변수만 거리에 들어간다.
query의 label/side를 임의값으로 바꿔도 결과가 같고, query를 단독 계산해도 동일함을 30개 fold에서 확인했다.
한 repeat에서 현재 validation fold 외 나머지 두 fold는 train이다. 이들의 train label 사용은 정상적인 OOF이며, 현재 validation fold의 다른 샘플을 거리 참조군에 넣지는 않는다.

## Export contract

- Unit: shot. 한 mold/repeat/shot_id당 1행. row_id는 의도적으로 빈 값이며 대표 제품 ID를 임의 지정하지 않는다.
- CN7: {meta['product_counts']['cn7']['unique_shots']} Shot ×5 = {meta['product_counts']['cn7']['records']} records.
- RG3: {meta['product_counts']['rg3']['unique_shots']} Shot ×5 = {meta['product_counts']['rg3']['records']} records.
- ambiguity min/max: {meta['minimum']:.16g} / {meta['maximum']:.16g}; missing ambiguity: 0; duplicate keys: 0.
- 모든 validation Shot이 매 repeat 정확히 한 값을 갖는다. 상세 fold별 Shot 수와 대응 제품 행 수는 *_counts.csv에 있다.
- row 단위 ablation에서는 원래 prediction에 mold/repeat/shot_id로 many-to-one join한 뒤 fold 일치를 검증하여 같은 Shot의 LH/RH에 같은 값을 붙일 수 있다. 제품별 다른 점수는 아니다. 원래 제품 y_true는 유지해야 한다.
- score가 높다는 것은 두 class까지의 거리가 비슷하다는 뜻이며, 불량 확률이나 직접적인 위험 점수가 아니다.
- one-side train Shot은 기존 정의대로 두 class 참조군에 동시에 들어간다. 이 Shot이 양쪽 최근접이면 ambiguity=1이 될 수 있다. 이를 바꾸면 새로운 방법이 되므로 그대로 유지했다.
- 기존 코드/config/모델/outputs {len(before)}개 파일 hash가 동일하다.

재현: `.venv\\Scripts\\python.exe -m src.analysis.export_ambiguity`
'''
    (OUT / f'{STEM}_README.md').write_text(note, encoding='utf-8')
    assert before == protected_hashes(), 'Protected files changed after export'
    print(json.dumps({k:meta[k] for k in ['train_fold_only','fold_scheme','unit','product_counts','minimum','maximum','missing_ambiguity','duplicate_keys','protected_files_unchanged']}, indent=2))


if __name__ == '__main__':
    main()
