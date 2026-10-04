"""Update E2 report/figures from saved detailed records only; no replay imports."""
import csv
import hashlib
import io
import os
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.audit_b_final_metadata import append_notes, SPC_NOTE, REASON_NOTE

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis'
SOURCE = ROOT / 'outputs/inspection/delay_replay_log_full.csv'
ARCHIVE = Path(os.environ.get('KAMP_DELIVERED_ARCHIVE', ROOT.parent / '제조데이터.zip'))  # 전달본 zip 위치(환경변수), 로컬 경로 하드코딩 금지
BASE = '제조데이터/kamp-injection-molding/outputs/analysis/'
REPORT = 'cn7_logreg_e2_shot1173_analysis.md'
Q1 = 'cn7_logreg_e2_q1_prediction_evidence.csv'
Q2 = 'cn7_logreg_e2_q2_inspection_evidence.csv'
KEYS = ['product', 'model', 'threshold', 'delay', 'cap_per10', 'rule', 't']
VALUES = ['score', 'thr', 'score_rank', 'score_pct', 'want_model', 'want_track', 'room', 'inspected_sides', 'inspected_parts', 'found']
UNCHANGED = ['cn7_e2_q3_lh_rh_evidence.csv', 'cn7_e2_q4_local_distance_evidence.csv',
             'fig_cn7_e2_q3_sameX_diffOutcome.png', 'fig_cn7_e2_q4_local_distance.png']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame):
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(str(v) for v in r) + ' |' for r in frame.itertuples(index=False, name=None)]])


def figures(target):
    colors = {'logreg': '#126e82', 'logreg_spc': '#727b89', 'spc_maxz': '#d0782d'}
    titles = {'logreg': 'Primary: Logistic', 'logreg_spc': 'Auxiliary: Logistic + SPC', 'spc_maxz': 'Auxiliary: SPC'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))
    for ax, model in zip(axes, ['logreg', 'logreg_spc', 'spc_maxz']):
        g = target[target.model.eq(model)]
        r = g.iloc[0]
        ax.plot([0, 1], [0, 0], color='#d7dce2', lw=5, zorder=1)
        ax.scatter([r.score], [0], s=140, color=colors[model], zorder=4)
        for q, offset in [('top2', .26), ('top5', -.34)]:
            t = g[g.threshold.eq(q)].iloc[0]
            ax.vlines(t.thr, -.12, .12, color='#333333', linestyle='--', lw=1.5)
            ax.annotate(f'{q} threshold = {t.thr:.5f}', (t.thr, 0), xytext=(min(t.thr, .77), offset),
                        arrowprops={'arrowstyle': '-', 'color': '#7e8790'}, ha='center', fontsize=10)
        ax.set_xlim(-.045, 1.045); ax.set_ylim(-.57, .62); ax.set_yticks([])
        ax.set_title(f'{titles[model]}  |  score {r.score:.5f}  |  rank {int(r.score_rank)}/1389  |  score_pct {r.score_pct:.4f}', loc='left', fontsize=12, weight='bold')
        lines = ', '.join(str(int(g[(g.threshold.eq(q)) & (g.delay.eq(0)) & (g.cap_per10.eq(2)) & (g.rule.eq('model'))].source_csv_line.iloc[0])) for q in ['top2', 'top5'])
        ax.text(0, -.50, f'Evidence: delay_replay_log_full.csv lines {lines}; delay=0, cap=2, rule=model', fontsize=9, color='#4b5563')
        ax.set_xlabel('Saved score / saved threshold (each model has its own score scale)')
    fig.suptitle('Figure 1 | CN7 shot1173: Logistic score below both inspection thresholds', x=.07, ha='left', fontsize=15, weight='bold')
    fig.text(.07, .015, 'Rank 1 = highest risk. Scores rounded to 5 decimals; thresholds are calibrated, not exact future-rank quotas.\nSPC driver: Average_Screw_RPM (z=-5.8903 after x10 correction). Recording-scale effects may remain; auxiliary only.', fontsize=9, color='#4b5563')
    fig.tight_layout(rect=[.04, .07, .99, .95], h_pad=1.8)
    fig.savefig(OUT/'fig_cn7_logreg_e2_q1_score_rank.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), gridspec_kw={'height_ratios': [1, 1, 1.7]})
    for ax, model in zip(axes, ['logreg', 'logreg_spc', 'spc_maxz']):
        ax.axis('off'); g = target[target.model.eq(model)]
        ax.set_title(titles[model], loc='left', fontweight='bold', color=colors[model])
        if model != 'spc_maxz':
            rows = []
            for cap in [2, 4]:
                a = g[g.cap_per10.eq(cap)]
                r = a[(a.threshold.eq('top2')) & (a.delay.eq(0)) & (a.rule.eq('model'))].iloc[0]
                rows.append([f'cap{cap}; all rules', '0 / 0', str(int(r.room)), 'none', 'MISS: no request', str(int(r.source_csv_line))])
        else:
            rows = []
            for cap, rules, label in [(2, ['model','model_track','model_rh','model_rh_track'], 'cap2; all rules'),
                                       (4, ['model','model_track'], 'cap4; model / model_track'),
                                       (4, ['model_rh','model_rh_track'], 'cap4; model_rh / model_rh_track')]:
                a = g[g.cap_per10.eq(cap) & g.rule.isin(rules)]
                r = a[a.threshold.eq('top2') & a.delay.eq(0)].iloc[0]
                rows.append([label, '1 / 0', str(int(r.room)), r.inspected_sides or 'none', 'FOUND' if r.found else 'MISS: room=0', str(int(r.source_csv_line))])
        tbl = ax.table(cellText=rows, colLabels=['Condition', 'want_model / track', 'room', 'Inspected sides', 'Observed result', 'CSV line*'],
                       cellLoc='left', colLoc='left', colWidths=[.26,.18,.07,.14,.23,.12], bbox=[0,.24,1,.61])
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#d7dce2');cell.set_facecolor('#e9f1f4' if r==0 else '#f8fafc')
            if r==0:cell.set_text_props(weight='bold')
        ax.text(0,.10,'Same pattern in both top2/top5 and delays 0/1/3/5. *Representative top2, delay=0 row; all rows in evidence CSV.',fontsize=9,color='#4b5563')
    fig.suptitle('Figure 2 | Selection miss vs capacity miss: direct E2 decision records', x=.05, ha='left', fontsize=15, weight='bold')
    fig.text(.05,.045,'Logistic: low score / low priority -> no request despite available room.\nSPC: recorded inspection outcomes only; RPM recording-scale effects limit process interpretation.\nSource: outputs/inspection/delay_replay_log_full.csv, CN7 t=1173; 64 conditions per model.',fontsize=11)
    fig.tight_layout(rect=[.035,.12,.98,.95],h_pad=2)
    fig.savefig(OUT/'fig_cn7_logreg_e2_q2_decision_flow.png',dpi=180);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_hash=sha(SOURCE)
    # Original artifacts are in the delivered archive, not the current checkout.
    with zipfile.ZipFile(ARCHIVE) as z:
        original=z.read(BASE+REPORT).decode('utf-8-sig')
        original_q34=original[original.index('## Q3.'):original.index('## 보조모델 비교')]
        for name in UNCHANGED:
            archived=z.read(BASE+name)
            if (OUT/name).exists():
                assert (OUT/name).read_bytes()==archived, f'Q3/Q4 changed: {name}'
            else:
                (OUT/name).write_bytes(archived)
    kept_hashes={name:sha(OUT/name) for name in UNCHANGED}
    data=pd.read_csv(SOURCE,keep_default_na=False)
    # Preserve actual physical CSV line numbers, with header on line 1.
    lines=[]
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);next(reader)
        for row in reader:lines.append(reader.line_num)
    assert len(lines)==len(data)
    data['source_csv_line']=lines
    assert not data.duplicated(KEYS).any()
    target=data[data['product'].eq('cn7') & data.t.eq(1173)].copy()
    assert set(target.model)=={'logreg','logreg_spc','spc_maxz'} and len(target)==192
    assert target.ts.eq('2020-11-03 04:44:17').all()
    assert target.defect_sides.eq('LH').all()
    groupkeys=['model','threshold','cap_per10','rule']
    assert target.groupby(groupkeys)[VALUES].nunique(dropna=False).eq(1).all().all()
    for model,g in target.groupby('model'):
        assert len(g)==64
        assert set(g.delay)=={0,1,3,5}
        assert set(g.threshold)=={'top2','top5'} and set(g.cap_per10)=={2,4}
        assert set(g.rule)=={'model','model_track','model_rh','model_rh_track'}
        assert g[['score','score_rank','score_pct']].nunique().eq(1).all()
        for _,full in data[data['product'].eq('cn7') & data.model.eq(model)].groupby(['threshold','delay','cap_per10','rule']):
            assert len(full)==1389 and set(full.t)==set(range(1389))
    assert np.allclose(target.score_pct,(target.score_rank/1389).round(4),rtol=0,atol=1e-12)
    logistic=target[target.model.eq('logreg')]
    hybrid=target[target.model.eq('logreg_spc')]
    spc=target[target.model.eq('spc_maxz')]
    # Expected interpretation is accepted only after all actual conditions pass.
    assert logistic.score.eq(0).all() and logistic.score_rank.eq(823).all()
    for g in [logistic,hybrid]:
        assert g.want_model.eq(0).all() and g.want_track.eq(0).all()
        assert g.score.lt(g.thr).all() and g.room.eq(g.cap_per10).all()
        assert g.inspected_sides.eq('').all() and g.inspected_parts.eq(0).all() and g.found.eq(0).all()
    assert hybrid.score.eq(.49665).all() and hybrid.score_rank.eq(832).all()
    assert spc.score.eq(.99289).all() and spc.score_rank.eq(30).all()
    assert spc.want_model.eq(1).all() and spc.want_track.eq(0).all() and spc.score.ge(spc.thr).all()
    caught=spc.cap_per10.eq(4)&spc.rule.isin(['model','model_track'])
    assert spc.loc[caught,'room'].eq(2).all() and spc.loc[caught,'inspected_sides'].eq('RH+LH').all()
    assert spc.loc[caught,'inspected_parts'].eq(2).all() and spc.loc[caught,'found'].eq(1).all()
    assert spc.loc[~caught,'room'].eq(0).all() and spc.loc[~caught,'inspected_sides'].eq('').all()
    assert spc.loc[~caught,'inspected_parts'].eq(0).all() and spc.loc[~caught,'found'].eq(0).all()
    target['source_file']='outputs/inspection/delay_replay_log_full.csv'
    target['source_sha256']=source_hash
    target['future_shots']=1389
    target['inspection_request']=(target.want_model.eq(1)|target.want_track.eq(1)).astype(int)
    target['conclusion']=np.where(target.model.ne('spc_maxz'),'prediction/selection miss: no request despite room',
                                 np.where(target.found.eq(1),'requested, RH+LH inspected, defect found','requested but room=0, not inspected'))
    logistic=target[target.model.eq('logreg')]
    logistic.to_csv(OUT/Q1,index=False,encoding='utf-8-sig')
    logistic.to_csv(OUT/Q2,index=False,encoding='utf-8-sig')
    target[target.model.ne('logreg')].to_csv(OUT/'cn7_e2_auxiliary_decision_evidence.csv',index=False,encoding='utf-8-sig')
    grouped=[]
    for keys,g in target.groupby(groupkeys,sort=False):
        r=dict(zip(groupkeys,keys));r.update({c:g[c].iloc[0] for c in VALUES})
        r.update(delays=','.join(map(str,sorted(g.delay))),source_csv_lines=','.join(map(str,g.sort_values('delay').source_csv_line)))
        grouped.append(r)
    grouped=pd.DataFrame(grouped);grouped.to_csv(OUT/'cn7_e2_detail_condition_evidence.csv',index=False,encoding='utf-8-sig')
    figures(target)
    prefix='''# 최종 CN7 Logistic 기준 E2 shot1173 오류분석

새 상세기록 `outputs/inspection/delay_replay_log_full.csv`의 CN7 `t=1173`, `ts=2020-11-03 04:44:17` 실제 행으로 Q1/Q2를 갱신했다. 모델 학습·추론·replay는 실행하지 않았다. 아래 CSV 행 번호는 헤더를 1행으로 센 물리적 행 번호다. Q3/Q4는 기존 직접 근거와 해석을 그대로 유지했다.

## Q1. Logistic은 shot1173 자체를 위험하게 평가했는가?

**Logistic은 낮은 score와 ranking 때문에 검사 요청을 만들지 않았다.** 모든 64조건에서 `score=0.00000`, `score_rank=823`, `score_pct=0.5925`이며 미래 1,389 Shot 중 823위다. `score_pct`는 rank/전체 Shot 수를 반올림한 값으로 여기서는 59.25%다. [근거: 원본 **90071행**(logreg/top2/delay0/cap2/model), **134519행**(logreg/top5/delay0/cap2/model)의 `score,score_rank,score_pct`; 전체64조건은 Q1 CSV의 `source_csv_line`.]

저장 threshold는 top2 **0.86303**, top5 **0.59979**이며 score가 모두 미달한다. [동일 두 행의 `thr,score,want_model=0`.]
`score=0.00000`은 저장 정밀도에서 약 0이라는 뜻이며, 반올림 전 값이 수학적으로 정확한 0이라는 뜻은 아니다. top2/top5는 보정 구간에서 정한 threshold 이름이므로 미래 rank의 정확한 상위 2/5% 할당과 동일시하지 않는다.

![Figure 1](fig_cn7_logreg_e2_q1_score_rank.png)

[Q1 실제 64행 근거 CSV](cn7_logreg_e2_q1_prediction_evidence.csv)

## Q2. Logistic 미포착은 capacity 문제인가, 요청 부재인가?

**검사 capacity가 남아 있었지만 요청이 발생하지 않은 예측/선정 단계 miss다.** 64조건 모두 `want_model=0`, `want_track=0`, `inspected_parts=0`, `inspected_sides` 빈 값, `found=0`이다. cap2는 `room=2`, cap4는 `room=4`로 남아 있다. [근거: **90071행**(logreg/top2/delay0/cap2/model), **95627행**(logreg/top2/delay0/cap4/model)의 해당 열; 아래 조건별 표는 delay0/1/3/5 전체 행 번호를 제공한다.]

따라서 Logistic의 미포착을 capacity 부족 또는 RH만 검사한 결과로 설명하지 않는다. `room>0`와 요청 플래그 0, 검사 수 0이 같은 실제 행에 함께 기록되어 있다. 미래 rank는 낮은 우선순위를 보여 주며, 직접적인 요청 판정은 score와 threshold 비교로 확인한다.

![Figure 2](fig_cn7_logreg_e2_q2_decision_flow.png)

[Q2 실제 64행 근거 CSV](cn7_logreg_e2_q2_inspection_evidence.csv) · [모델별 전체 조건과 원본 행 번호](cn7_e2_detail_condition_evidence.csv)

'''
    auxiliary='''## 보조모델 비교 — Logistic과 구분

- **logreg_spc:** 64조건 모두 `score=0.49665`, rank **832/1389**, `score_pct=0.5990`. top2 `thr=0.96232`, top5 `thr=0.88876`보다 낮고 `want_model=want_track=0`. cap2/4의 room은 각각 2/4이며 무검사·미포착이다. [**1175행**(top2/delay0/cap2/model), **45623행**(top5/delay0/cap2/model), **6731행**(top2/delay0/cap4/model)의 `score,score_rank,score_pct,thr,want_model,want_track,room,inspected_sides,found`.]
- **SPC:** 64조건 모두 `score=0.99289`, rank **30/1389**, `score_pct=0.0216`, `want_model=1`, `want_track=0`. top2 `thr=0.98910`, top5 `thr=0.97679`를 넘어서 검사 요청이 있었다. [**178967행**(top2/delay0/cap2/model), **223415행**(top5/delay0/cap2/model)의 해당 열.]
- **SPC cap2:** 모든 rule에서 `room=0`, `inspected_parts=0`, side 빈 값, `found=0`. 요청 후 capacity가 0인 조건에서 미포착이다. [대표 **178967행**, 모든 조건은 아래 행 번호표.]
- **SPC cap4 model/model_track:** `room=2`, `inspected_sides=RH+LH`, `inspected_parts=2`, `found=1`로 포착했다. [**184523행**, **185912행**(top2/delay0)의 해당 열.]
- **SPC cap4 model_rh/model_rh_track:** `room=0`, 검사 수0·side 빈 값·`found=0`으로 미포착이다. [**187301행**, **188690행**(top2/delay0)의 해당 열.] 이 사례는 RH만 검사한 miss가 아니라 무검사다.

SPC는 총 64조건 중 16조건 포착, 48조건 미포착이다. 두 threshold와 네 delay 모두 같은 조건별 패턴임을 직접 대조했다. [아래 SPC 16개 조건 그룹 × delay 4개, 각 `found` 열.]

[보조모델 실제 128행 근거 CSV](cn7_e2_auxiliary_decision_evidence.csv)

## 조건별 직접 근거

아래 각 행은 threshold/cap/rule별 묶음이다. 표시된 값은 delay **0,1,3,5 네 행 모두 동일**함을 검증했다. `source_csv_lines`는 이 delay 순서에 대응한다. side 빈 값은 원본 그대로이며 검사 수 0과 함께 무검사를 뜻한다. 전체 원본 열은 근거 CSV에 보존했다.

'''
    show=grouped.copy();show['inspected_sides']=show.inspected_sides.replace('','(empty)')
    suffix='''

## 마지막 요약

| 질문 | 직접 관측 | 판정 |
|---|---|---|
| Q1 위험평가 | 약0, rank823/1389, threshold 미달 (90071·134519행) | 낮은 우선순위/점수로 요청 없음 |
| Q2 검사운영 | 요청0, room2/4, 무검사 (90071·95627행) | capacity 부족이 아닌 예측/선정 단계 miss |
| Q3 관측정보 | 기존 동일23X, LH미성형/RH정상 근거 유지 | side별 outcome 구분정보 미관측 |
| Q4 공정이탈 | 기존 event0 3.56–41.63 / 대상0.688,1.072 근거 유지 | 같은 큰 국소이탈 가설 미지지 |

현재 상세 기록으로 Q1/Q2 결론을 직접 확인할 수 있으므로 추가 학습 또는 replay 재실행이 필요하지 않다. Q3/Q4의 사후 해석을 Logistic score의 원인으로 단정하지 않는다.
'''
    content=prefix+original_q34+auxiliary+table(show)+suffix
    content=content.replace('## 보조모델 비교 — Logistic과 구분',
        '## 보조모델 비교 — Logistic과 구분\n\nSPC 최대 이탈 변수는 ×10 보정 후에도 Average_Screw_RPM이다. 아래 포착·미포착은 저장된 검사운영 결과이며, 기록 눈금/정밀도 변화의 영향 때문에 불량의 직접 공정 신호를 탐지했다는 근거로 해석하지 않는다. 상세 driver 검증은 문서 마지막 절에 기록했다.')
    (OUT/REPORT).write_bytes(content.encode('utf-8'))
    actual=(OUT/REPORT).read_bytes().decode('utf-8')
    assert actual[actual.index('## Q3.'):actual.index('## 보조모델 비교')]==original_q34
    append_notes(OUT/REPORT,SPC_NOTE,REASON_NOTE)
    assert all(sha(OUT/n)==h for n,h in kept_hashes.items())
    assert sha(SOURCE)==source_hash
    meta=dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=source_hash,source_lines='1-based physical CSV lines, header=1',
              target_rows=192,conditions_per_model=64,future_cn7_shots_per_condition=1389,all_conditions_verified=True,
              source_archive=ARCHIVE.name,original_q34_sections_preserved=True,q34_sha256=kept_hashes,
              training_executed=False,replay_executed=False,source_csv_unchanged=True)
    (OUT/'cn7_e2_detail_update_verification.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(meta,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
