import os
from pathlib import Path

import pytest

from medical_agent.rerank_provider import build_rerank_provider


DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def _load_dotenv_if_present() -> None:
    env_path = Path("/root/llm_learning/.env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_rerank_model() -> str:
    return os.environ.get("RERANK_MODEL", "").strip() or DEFAULT_RERANK_MODEL


def _has_smoke_config() -> bool:
    _load_dotenv_if_present()
    return bool(_resolve_rerank_model())


@pytest.mark.skipif(
    not _has_smoke_config(),
    reason="Smoke test requires RERANK_MODEL or default rerank model to be available",
)
def test_rerank_provider_smoke() -> None:
    """真实 smoke test：初始化 reranker，并对一小组候选进行排序。"""
    model_name = _resolve_rerank_model()
    provider = build_rerank_provider(model_name=model_name)

    query = "在给小儿喂药时，有哪些关键的使用方法和禁忌？"
    documents = [
        """题目：复方丹参片的使用注意事项
正确选项：ACD
解析：本题考查复方丹参片的注意事项。复方丹参片的使用注意事项孕妇慎用（孕妇慎用正确）、脾胃虚寒者慎用（脾胃虚寒者慎用正确）、肝肾功能异常者慎用（肝肾功能异常者慎用正确）。复方丹参片（丸、胶囊、滴丸）【注意事项】对本品及所含成分过敏者禁用。过敏体质者慎用。孕妇慎用。寒凝血瘀胸痹心痛者不宜使用，脾胃虚寒者慎用。服药期间，忌食生冷、辛辣、油腻食物，忌烟、酒、浓茶。治疗期间，如心绞痛持续发作，宜加用硝酸酯类药。如果出现剧烈心绞痛、心肌梗死等，应及时送医院救治。个别人服药后胃脘不适，宜饭后服用。肝肾功能异常者慎用。""",
        """
题目：非处方药的使用要求
正确选项：ABCDE
解析：本题考查非处方药的使用注意。非处方药的使用要求老年人、儿童不宜自行选择用药（老年人、儿童不宜自行选择用药正确）、不可超量或过久服用（不可超量或过久服用正确）、如症状未见减轻或缓解应及时就医（如症状未见减轻或缓解应及时就医正确）、严格按说明书用药（严格按说明书用药正确）、可在执业药师指导下选用药品（E对）。
""",
        """
题目：下列关于小儿用药特点的描述，错误的是
正确选项：婴幼儿鼻饲给药是不安全的
解析：本题考查新生儿用药。下列关于小儿用药特点的描述，错误的是婴幼儿鼻饲给药是不安全的（婴幼儿鼻饲给药是不安全的错误，为本题正确答案）。口服给药是最方便、最安全、最经济的给药途径，但影响因素较多，剂量不如注射给药准确，特别是吞咽能力差的婴幼儿受到一定限制。幼儿用糖浆、水剂、冲剂等较合适，年长儿可用片剂或丸剂，服药时要注意避免牛奶、果汁等食物的影响；小婴儿喂药时最好将其抱起或使头略抬高，以免呛咳时将药吐出。病情需要时可采用鼻饲给药。注射给药比口服给药起效快，但对小儿刺激大（皮下注射给药不适于新生儿正确）。临床上肌内注射部位多选择在臀大肌外上方，但注射次数过多可能造成臀部肌肉损害，需加以注意，且不得用于儿童（早产儿皮肤薄不宜肌内注射正确）。婴幼儿的液体容量小，输液速度不能过快（婴幼儿静脉给药速度不要过快正确）。婴幼儿角质层薄，且相对表面积大，透皮吸收容易导致中毒（婴幼儿使用透皮吸收药物容易中毒正确）。
""",
        """
解析：本题考查新生儿用药。口服给药是最方便、最安全、最经济的给药途径，但影响因素较多，剂量不如注射给药准确，特别是吞咽能力差的婴幼儿受到一定限制。幼儿用糖浆、水剂、冲剂等较合适（口服给药时以糖浆剂为宜正确），年长儿可用片剂或丸剂，服药时要注意避免牛奶、果汁等食物的影响；小婴儿喂药时最好将其抱起或使头略抬高，以免呛咳时将药吐出。病情需要时可采用鼻饲给药。新生儿血脑屏障发育不全，通透性高，很多药物易通过血脑屏障，使中枢神经系统易受药物影响。如吗啡较易使新生儿呼吸中枢受抑制（使用吗啡易引起呼吸抑制正确）；抗组胺药、氨茶碱、阿托品可致昏迷或惊厥（使用氨茶碱无兴奋神经系统作用错误）。应用中枢镇静药时年龄愈小耐受力愈大（应用中枢镇静药时年龄愈小耐受力愈大正确）。肌内注射时药物的吸收与局部血流量有关（肌内注射给药不影响吸收错误），要充分考虑注射部位的吸收状况，避免局部结块、坏死，如使用含苯甲醇为添加剂的溶媒会导致臀肌挛缩症的严重不良反应；临床上肌内注射部位多选择在臀大肌外上方，但注射次数过多可能造成臀部肌肉损害，需加以注意，且不得用于儿童。
""",
    ]

    results = provider.rerank(query, documents, top_k=3)

    print(f"model: {model_name}")
    print(f"query: {query}")
    for rank, item in enumerate(results, 1):
        print("-" * 72)
        print(f"rank: {rank}")
        print(f"index: {item.index}")
        print(f"score: {item.score:.4f}")
        print(f"document: {item.document}")

    assert len(results) == 3
    assert all(isinstance(item.score, float) for item in results)
    assert results[0].score >= results[-1].score
