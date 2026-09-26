"""공개 원본 검증으로 확인한 사실 (인용용 상수).

출처: scripts/verify_public_link.py --enable (KAMP 공개 원본 zip, 읽기 전용, 2026-09-26 실행)
→ outputs/verification/public_link_summary.json. 파이프라인은 공개 원본을 읽지 않고 이 수치를 설명에만 인용한다.
"""

SOURCE = "scripts/verify_public_link.py (KAMP 공개 원본, 읽기 전용 검증, 2026-09-26)"

PUBLIC_VERIFIED = {
    "cn7": {
        "first_row_RH_share": 0.658,             # 짝의 앞 행(row_id 작은 쪽)이 RH 인 비율
        "one_sided_defect_first": (10, 11),      # 한쪽만 불량인 짝에서 불량 행이 앞 행인 수 / 짝 수
        "expected_first_if_cavity": 0.629,       # 순서가 캐비티를 뜻한다면 기대되는 비율
        "labeled_rows_in_unlabeled": (1210, 1211),
        "labeled_defects_in_unlabeled": 17,
    },
    "rg3": {
        "first_row_RH_share": 0.563,
        "one_sided_defect_first": (23, 25),
        "expected_first_if_cavity": 0.563,
        "labeled_rows_in_unlabeled": (774, 1182),
        "labeled_defects_in_unlabeled": 16,
    },
}


def pair_order_leak_text() -> str:
    parts = []
    for p, f in PUBLIC_VERIFIED.items():
        k, n = f["one_sided_defect_first"]
        parts.append(f"{p.upper()} 앞 행=RH {f['first_row_RH_share']:.1%}, 한쪽 불량 짝의 불량 행이 앞 행 {k}/{n}"
                     f"({k / n:.1%}, 캐비티라면 기대 {f['expected_first_if_cavity']:.1%})")
    return "; ".join(parts)


def overlap_text() -> str:
    return ", ".join(f"{p.upper()} {f['labeled_rows_in_unlabeled'][0]}/{f['labeled_rows_in_unlabeled'][1]}행"
                     for p, f in PUBLIC_VERIFIED.items())
