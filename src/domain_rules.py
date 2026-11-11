"""Korean household waste-sorting domain rules & probability-aware decision engine.

Maps the 9 RealWaste classes + model probability distributions (top-k margins,
normalized Shannon entropy, and co-occurring contamination probabilities) into
actionable disposal guidance aligned with the Korean Ministry of Environment's
'내 손안의 분리배출' guidelines. No external LLM/API is called.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math

CLASS_NAMES: list[str] = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

RECYCLABLE_CLASSES: set[str] = {
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Textile Trash",
}


@dataclass(frozen=True)
class WasteRule:
    en_name: str
    ko_name: str
    bin_category: str
    bin_badge_type: str  # recyclable | food | general | special
    recyclable_status: str
    steps: list[str]
    penalty_trap: str
    examples: str


KOREAN_WASTE_RULES: dict[str, WasteRule] = {
    "Cardboard": WasteRule(
        en_name="Cardboard",
        ko_name="골판지·택배 상자",
        bin_category="종이·골판지류 전용 수거함",
        bin_badge_type="recyclable",
        recyclable_status="재활용 가능 (이물질 제거 필수)",
        steps=[
            "상자에 붙은 박스테이프, 운송장 스티커(감열지), 철핀을 완전히 제거합니다.",
            "부피를 줄이기 위해 상자를 납작하게 펼쳐 접어 묶음 배출합니다.",
            "기름·음식물 국물이 배어든 피자박스/치킨박스 속지는 떼어내 일반 종량제 봉투에 버립니다.",
        ],
        penalty_trap="운송장 스티커와 비닐 코팅 테이프를 떼지 않고 배출하면 제지 공정 불량 및 수거 거부 사유가 됩니다.",
        examples="택배 박스, 골판지 포장재, 계란 종이판(오염 없는 것), 골판지 완충재",
    ),
    "Food Organics": WasteRule(
        en_name="Food Organics",
        ko_name="음식물류 폐기물",
        bin_category="음식물 전용 종량제 봉투 / RFID 수거함",
        bin_badge_type="food",
        recyclable_status="음식물 사료·퇴비화 대상",
        steps=[
            "물기를 최대한 짜서 수분을 제거한 뒤 음식물 전용 수거함·봉투에 배출합니다.",
            "비닐, 이쑤시개, 병뚜껑, 일회용 나무젓가락 등 이물질이 섞이지 않도록 분리합니다.",
            "염분이 많은 국물·장류는 건더기만 걸러서 배출합니다.",
        ],
        penalty_trap="동물이 먹을 수 없는 딱딱한 뼈다귀, 조개·달걀 껍데기, 과일 씨앗(복숭아·아보카도 씨)은 음식물이 아닌 일반 종량제로 배출해야 합니다.",
        examples="남은 밥·반찬, 과일 껍질(부드러운 것), 채소 자투리, 빵·면류 잔여물",
    ),
    "Glass": WasteRule(
        en_name="Glass",
        ko_name="유리병·유리 용기",
        bin_category="유리병류 수거함 (깨진 유리는 불연성 마대)",
        bin_badge_type="recyclable",
        recyclable_status="재활용 가능 (내열유리·파손유리 제외)",
        steps=[
            "내용물을 깨끗이 비우고 물로 헹군 뒤 금속·플라스틱 뚜껑을 분리합니다.",
            "담배꽁초나 이물질을 병 안에 넣지 않고 깨지지 않게 수거함에 넣습니다.",
            "소주병·맥주병 등 빈용기보증금 대상 병은 편의점·마트에 반납해 보증금을 환급받으세요.",
        ],
        penalty_trap="깨진 유리, 도자기·사기그릇, 내열유리(전자레인지용 용기), 거울은 절대 유리병 수거함에 넣으면 안 되며 신문지에 싸서 종량제 또는 불연성 마대로 배출해야 합니다.",
        examples="음료수 병, 맥주·소주병, 잼·소스 유리병, 화장품 유리 용기",
    ),
    "Metal": WasteRule(
        en_name="Metal",
        ko_name="금속캔·고철류",
        bin_category="캔·고철류 전용 수거함",
        bin_badge_type="recyclable",
        recyclable_status="재활용 우수 품목",
        steps=[
            "음료·통조림 캔은 내용물을 비우고 물로 헹군 뒤 가능하면 발로 밟아 압착합니다.",
            "부탄가스·살충제·헤어스프레이 등 압력 용기는 통풍이 잘되는 야외에서 구멍을 뚫어 잔여 가스를 완전히 제거합니다.",
            "플라스틱 뚜껑이나 고무 패킹이 결합된 경우 분리하여 각각 배출합니다.",
        ],
        penalty_trap="잔여 가스가 남은 부탄가스·스프레이 캔을 구멍 뚫지 않고 배출하면 압축 수거 차량 화재·폭발 사고의 직접 원인이 됩니다.",
        examples="알루미늄 음료캔, 통조림캔, 부탄가스캔, 알루미늄 호일(깨끗한 것), 공구·철사 고철",
    ),
    "Miscellaneous Trash": WasteRule(
        en_name="Miscellaneous Trash",
        ko_name="복합재질·오염 일반쓰레기",
        bin_category="일반 종량제 봉투",
        bin_badge_type="general",
        recyclable_status="재활용 불가 (소각·매립 대상)",
        steps=[
            "음식물·기름으로 심하게 오염되어 세척이 불가능한 비닐·종이·용기는 일반 종량제 봉투에 넣습니다.",
            "여러 재질이 강하게 접착된 복합소재(영수증 감열지, 보온·보냉팩, 칫솔, 고무장갑 등)도 종량제 봉투로 배출합니다.",
            "날카로운 조각이 포함된 경우 종이로 감싸 환경미화원 부상을 예방합니다.",
        ],
        penalty_trap="오염된 일회용기나 영수증·물티슈를 재활용함에 섞어 버리면 전체 재활용 수거물이 오염되어 과태료(최대 10~30만 원) 부과 대상이 됩니다.",
        examples="오염된 포장재, 영수증(감열지), 사용한 휴지·물티슈, 빨대, 기저귀, 복합재질 쓰레기",
    ),
    "Paper": WasteRule(
        en_name="Paper",
        ko_name="일반 종이·인쇄물",
        bin_category="종이류 수거함 (종이팩은 별도 배출)",
        bin_badge_type="recyclable",
        recyclable_status="재활용 가능 (비닐코팅·감열지 제외)",
        steps=[
            "물기나 음식물이 묻지 않도록 하고 구겨지지 않게 펴서 차곡차곡 묶어 배출합니다.",
            "노트·책자의 스프링, 비닐 표지, 스테이플러 심을 제거한 뒤 배출합니다.",
            "우유팩·두유팩(살균팩/멸균팩)은 일반 폐지와 펄프 재질이 다르므로 헹궈서 펼쳐 말린 뒤 종이팩 전용 수거함에 배출합니다.",
        ],
        penalty_trap="영수증(감열지), 금박·은박지, 비닐 코팅 전단지, 오염된 핸드타월·휴지는 종이류가 아닌 일반 종량제 봉투에 버려야 합니다.",
        examples="신문지, 복사용지, 책·잡지, 노트, 종이 쇼핑백(코팅·끈 제거)",
    ),
    "Plastic": WasteRule(
        en_name="Plastic",
        ko_name="플라스틱 용기·페트류",
        bin_category="투명페트병 전용함 / 플라스틱류 수거함",
        bin_badge_type="recyclable",
        recyclable_status="재활용 가능 (비우기·헹구기·비닐제거)",
        steps=[
            "생수·음료 투명 페트병은 내용물을 비우고 비닐 라벨을 떼어낸 뒤 압착하여 '투명페트병 전용 수거함'에 별도 배출합니다.",
            "유색 플라스틱·배달용기는 음식물·양념을 깨끗이 씻어낸 뒤 일반 플라스틱 수거함에 넣습니다.",
            "샴푸·세제 용기의 펌프(금속 스프링 포함)는 분리하여 일반쓰레기로 버리고 본체만 배출합니다.",
        ],
        penalty_trap="투명 페트병을 일반 유색 플라스틱과 혼합 배출하거나, 씻지 않은 마라탕·떡볶이 빨간 기름 자국 용기를 그대로 배출하면 재활용 선별에서 탈락합니다.",
        examples="투명 생수·음료 페트병, 세제·샴푸 통, 플라스틱 배달용기, 테이크아웃 컵",
    ),
    "Textile Trash": WasteRule(
        en_name="Textile Trash",
        ko_name="폐의류·섬유류",
        bin_category="의류수거함 (오염·솜류는 종량제/대형폐기물)",
        bin_badge_type="special",
        recyclable_status="상태에 따라 조건부 재활용",
        steps=[
            "세탁되어 깨끗하고 다시 입을 수 있는 옷·가방·신발(짝 맞음)은 건조한 상태로 의류수거함에 넣습니다.",
            "젖었거나 기름·페인트·곰팡이로 오염된 옷, 걸레, 양말·속옷류는 의류수거함이 아닌 일반 종량제 봉투에 배출합니다.",
            "솜이불, 베개, 방석, 캐리어, 바퀴 달린 신발은 의류수거함 금지 품목이므로 종량제 봉투 또는 대형폐기물 스티커를 부착해 배출합니다.",
        ],
        penalty_trap="젖은 수건·걸레·솜이불을 의류수거함에 넣으면 수거함 전체 의류에 곰팡이가 번져 전량 폐기되며 지자체 단속 대상이 됩니다.",
        examples="헌 옷, 직물 조각, 가방·모자(재사용 가능 시 의류수거함 / 훼손·오염 시 종량제)",
    ),
    "Vegetation": WasteRule(
        en_name="Vegetation",
        ko_name="초목류·식물 잔재물",
        bin_category="일반 종량제 봉투 (음식물쓰레기 절대 금지!)",
        bin_badge_type="general",
        recyclable_status="음식물 사료화 불가 → 일반쓰레기",
        steps=[
            "화분에서 죽은 식물, 나뭇가지, 낙엽, 생화 꽃다발은 흙을 털어낸 뒤 일반 종량제 봉투에 버립니다.",
            "부피가 큰 나뭇가지는 30cm 이하로 잘라 묶거나 대형·초목류 전용 마대로 배출합니다.",
            "양파·마늘·옥수수 마른 겉껍질, 대파·쪽파 뿌리 역시 섬유질이 질겨 가축 사료화가 불가능하므로 일반 종량제 봉투에 배출합니다.",
        ],
        penalty_trap="[자취생 과태료 1위 주의] 식물 잎·줄기·뿌리·낙엽을 '유기물이니까 음식물쓰레기겠지'라고 착각해 음식물 수거함에 버리면 분쇄기 고장 및 과태료 부과 대상입니다!",
        examples="화분 식물·줄기·뿌리, 낙엽, 잔디·잡초, 시든 꽃다발, 마른 채소 껍질·뿌리",
    ),
}


def build_decision_advice(probs_by_class: dict[str, float]) -> dict[str, Any]:
    """Generate structured disposal recommendation & warnings from class probabilities."""
    ranked = sorted(probs_by_class.items(), key=lambda kv: kv[1], reverse=True)
    top1_cls, top1_prob = ranked[0]
    top2_cls, top2_prob = ranked[1]
    margin = top1_prob - top2_prob

    # Normalized Shannon entropy in [0, 1]
    eps = 1e-12
    entropy = -sum(p * math.log(max(p, eps)) for _, p in ranked)
    max_entropy = math.log(len(ranked))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    rule = KOREAN_WASTE_RULES[top1_cls]
    alerts: list[dict[str, str]] = []

    # 1) Out-of-distribution / low confidence check
    is_uncertain = top1_prob < 0.42 or norm_entropy > 0.74
    if is_uncertain:
        alerts.append(
            {
                "level": "warning",
                "title": "낮은 판독 신뢰도 (재촬영 권장)",
                "message": (
                    f"최고 확률이 {top1_prob * 100:.1f}%(정규화 엔트로피 {norm_entropy:.2f})로 낮습니다. "
                    "여러 폐기물이 겹쳐 있거나 학습 도메인 외 물체일 수 있으니 단일 품목을 밝은 조명 아래 중앙에 두고 촬영해 주세요."
                ),
            }
        )

    # 2) Contamination co-occurrence check on recyclables
    contamination_prob = probs_by_class.get("Miscellaneous Trash", 0.0) + probs_by_class.get(
        "Food Organics", 0.0
    )
    contamination_suspected = (
        top1_cls in RECYCLABLE_CLASSES and contamination_prob >= 0.18
    )
    if contamination_suspected:
        alerts.append(
            {
                "level": "caution",
                "title": "표면 오염·이물질 혼입 주의 감지",
                "message": (
                    f"주 재질은 '{rule.ko_name}'({top1_prob * 100:.1f}%)이나 오염·음식물 연관 클래스 확률 합계가 "
                    f"{contamination_prob * 100:.1f}%로 동반 감지되었습니다. 물 세척으로 이물질이 지워지지 않으면 "
                    "'일반 종량제 봉투'로 배출하세요."
                ),
            }
        )

    # 3) Close boundary disambiguation alert
    if margin < 0.20 and not is_uncertain:
        second_rule = KOREAN_WASTE_RULES[top2_cls]
        alerts.append(
            {
                "level": "info",
                "title": f"유사 품목 경계 확인 ({rule.ko_name} vs {second_rule.ko_name})",
                "message": (
                    f"1순위 '{rule.ko_name}'({top1_prob * 100:.1f}%)와 2순위 '{second_rule.ko_name}'({top2_prob * 100:.1f}%)의 "
                    f"확률 격차({margin * 100:.1f}%p)가 좁습니다. 두 품목의 배출 기준을 함께 확인하세요."
                ),
            }
        )

    # 4) Special high-impact Korean sorting trap highlights
    if top1_cls == "Vegetation":
        alerts.append(
            {
                "level": "critical",
                "title": "음식물쓰레기 오배출 주의 (초목류)",
                "message": "식물 잎·줄기·화분 잔재·뿌리는 가축 사료화가 불가하여 음식물 수거함 배출 시 과태료 대상입니다. 반드시 일반 종량제 봉투에 버리세요!",
            }
        )
    elif top1_cls == "Textile Trash":
        alerts.append(
            {
                "level": "caution",
                "title": "건조·오염 여부 확인 필수 (섬유류)",
                "message": "깨끗하고 마른 의류만 의류수거함에 배출 가능하며, 젖거나 오염된 천·걸레·솜이불은 일반 종량제 봉투(또는 대형폐기물)로 배출해야 합니다.",
            }
        )

    return {
        "predicted_class": top1_cls,
        "predicted_ko_name": rule.ko_name,
        "confidence": round(top1_prob, 4),
        "margin_top1_top2": round(margin, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "contamination_score": round(contamination_prob, 4),
        "is_uncertain": is_uncertain,
        "contamination_suspected": contamination_suspected,
        "rule": asdict(rule),
        "alerts": alerts,
        "top_predictions": [
            {
                "class_en": cls_name,
                "class_ko": KOREAN_WASTE_RULES[cls_name].ko_name,
                "bin_category": KOREAN_WASTE_RULES[cls_name].bin_category,
                "probability": round(prob, 4),
            }
            for cls_name, prob in ranked
        ],
    }
