"""Read-only source audit of CN7 2020-10-27 defects; no model execution."""
import json
from pathlib import Path
import pandas as pd
import _public

ROOT=Path(__file__).resolve().parents[1]


def timeline(times, threshold=300):
    s=pd.DataFrame({'TimeStamp':pd.to_datetime(times).drop_duplicates().sort_values().to_numpy()})
    s['previous_timestamp']=s.TimeStamp.shift()
    s['gap_sec']=s.TimeStamp.diff().dt.total_seconds()
    s['restart_candidate']=s.gap_sec.gt(threshold)
    s['episode']=s.restart_candidate.cumsum()
    s['shot_since_candidate']=s.groupby('episode').cumcount()+1
    return s


def main():
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
    src=_public.public_path(cfg)
    raw=_public.read_csv(src,'labeled_data.csv')
    raw['TimeStamp']=pd.to_datetime(raw.TimeStamp)
    cn=raw[raw.PART_NAME.astype(str).str.contains('CN7',case=False)].copy()
    day=cn[cn.TimeStamp.dt.strftime('%Y-%m-%d').eq('2020-10-27')].copy()
    feats=[c for c in pd.read_csv(ROOT/'data/raw/moldset_labeled_cn7.csv',nrows=0).columns if c not in ['Unnamed: 0','PassOrFail','Clamp_Open_Position']]
    assert day.groupby('TimeStamp')[feats].nunique(dropna=False).le(1).all().all()
    assert len(day.drop_duplicates(feats))==day.TimeStamp.nunique()
    assert day.groupby(['TimeStamp','PART_NAME'])[['PassOrFail','Reason']].nunique(dropna=False).le(1).all().all()
    parts=day.drop_duplicates(['TimeStamp','PART_NAME']).copy()
    parts['cavity']=parts.PART_NAME.str.extract(r'\b(LH|RH)\b',expand=False)
    assert parts.cavity.notna().all() and set(parts.PassOrFail)<= {'Y','N'}
    shots=timeline(day.TimeStamp)
    shots['day_shot_number']=range(1,len(shots)+1)
    defects=parts[parts.PassOrFail.eq('N')]
    detail=[]
    for t,g in defects.groupby('TimeStamp'):
        row=shots[shots.TimeStamp.eq(t)].iloc[0].to_dict()
        row.update(reason=';'.join(sorted(g.Reason.dropna().unique())),cavity=';'.join(sorted(g.cavity.unique())),
                   defect_type='both-side' if set(g.cavity)=={'LH','RH'} else 'one-side',
                   all_initial_allowance=bool(g.Reason.eq('초기허용불량').all()),inspection_rule_match='unavailable')
        detail.append(row)
    detail=pd.DataFrame(detail)
    subset=_public.read_csv(src,'moldset_labeled.csv')
    sub=subset[subset.PART_NAME.str.contains('CN7') & subset.TimeStamp.str.startswith('2020-10-27')]
    assert set(pd.to_datetime(sub[sub.PassOrFail.eq(1)].TimeStamp))==set(defects.TimeStamp)
    # Same-machine observations from both raw tables distinguish CN7 gaps from other-product activity.
    unl=_public.read_csv(src,'unlabeled_data.csv',usecols=['TimeStamp','PART_NAME','EQUIP_NAME'])
    eq=day.EQUIP_NAME.unique(); assert len(eq)==1
    observed=pd.concat([raw[['TimeStamp','PART_NAME','EQUIP_NAME']],unl],ignore_index=True)
    observed['TimeStamp']=pd.to_datetime(observed.TimeStamp)
    machine=timeline(observed[observed.EQUIP_NAME.eq(eq[0])].TimeStamp)
    gaps=shots[shots.restart_candidate].copy()
    machine_prev=machine.set_index('TimeStamp')
    gaps['machine_gap_sec']=gaps.TimeStamp.map(machine_prev.gap_sec)
    sensitivity=[]
    for threshold in [180,300,600]:
        ts=timeline(day.TimeStamp,threshold).set_index('TimeStamp')
        for t in detail.TimeStamp:
            r=ts.loc[t]
            sensitivity.append(dict(threshold_sec=threshold,TimeStamp=str(t),shot_since_candidate=int(r.shot_since_candidate),episode=int(r.episode)))
    out=ROOT/'outputs/analysis/audit_oct27'; out.parent.mkdir(parents=True,exist_ok=True)
    parts[['TimeStamp','PART_NAME','PassOrFail','Reason','cavity']].to_csv(f'{out}_parts.csv',index=False)
    shots.to_csv(f'{out}_timeline.csv',index=False)
    detail.to_csv(f'{out}_defects.csv',index=False)
    gaps.to_csv(f'{out}_gaps.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(f'{out}_gap_sensitivity.csv',index=False)
    first=machine_prev.loc[shots.TimeStamp.iloc[0]]
    summary=dict(source=str(src),primary_member='dataset/labeled_data.csv',raw_rows=len(day),unique_parts=len(parts),
                 observed_shots=len(shots),defect_shots=len(detail),defect_parts=len(defects),subset_shots=sub.TimeStamp.nunique(),
                 all_defects_initial_allowance=bool(detail.all_initial_allowance.all()),gap_threshold_sec=300,
                 normal_gap_median_sec=float(shots.gap_sec.median()),equipment=eq[0],
                 first_observed_time=str(shots.TimeStamp.iloc[0]),last_observed_time=str(shots.TimeStamp.iloc[-1]),
                 preceding_machine_observation=str(first.previous_timestamp),preceding_machine_gap_sec=float(first.gap_sec),
                 inspection_rule_matching='Incomplete: teammate output unavailable; not simulated',
                 limitations=['Observed gaps are not proof of machine downtime; source coverage may be incomplete',
                              '300-second restart-candidate threshold is an audit convention; 180/600 sensitivity saved',
                              'First daily observation is left-censored, not an asserted restart'])
    Path(f'{out}_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(detail.to_string(index=False)); print(gaps.to_string(index=False))


if __name__=='__main__': main()
