# %% [markdown]
# # 한국어 가짜뉴스·풍자뉴스 2단계 판독 파이프라인
#
# 이 파일은 Jupyter Notebook에서 셀 단위로 실행하거나 일반 Python 모듈로 불러올 수 있습니다.
#
# 핵심 구조
# 1. Stage 1: 경량 한국어 BERT 계열 모델로 `진짜` 대 `의심(가짜+풍자)` 확률 추정
# 2. Stage 2-A: Bi-LSTM으로 감정자극 단어와 과장 표현 단어를 검출하고 어절 비율 계산
# 3. Stage 2-B: 경량 한국어 BERT 계열 모델로 아이러니 지수(0~1) 추정
# 4. 규칙 기반 판정: 과장+아이러니가 높거나 아이러니가 특히 높으면 풍자, 나머지는 가짜
#
# 주의: LLM이 부여한 "attention"은 모델의 실제 self-attention 정답이 아닙니다.
# 이 구현에서는 이를 중요한 근거 구간을 표시하는 `rationale supervision`으로 해석합니다.

# %%
from __future__ import annotations

PIPELINE_VERSION = "2026.07.18-stage2-missing-span-fix"


import ast
import bisect
import copy
import inspect
import json
import math
import os
import random
import re
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("once")


# %% [markdown]
# ## 1. 설정

# %%
@dataclass
class Config:
    # 데이터 파일. 사용자가 설명한 `|` 구분 형식을 기본값으로 둡니다.
    data_path: str = "news_dataset.csv"
    delimiter: str = "|"

    # CSV 문자 인코딩입니다.
    # "auto"이면 utf-8-sig → utf-8 → cp949 → euc-kr 순서로 자동 시도합니다.
    encoding: str = "auto"

    # 음수 irony_score 처리 방식:
    # - "missing": 미주석값(NaN)으로 처리하고 아이러니 모델 학습에서 제외
    # - "zero": 아이러니 없음(0.0)으로 처리
    # - "rescale_minus1_1": 전체 -1~1 척도를 0~1로 변환
    # - "error": 음수가 있으면 오류 발생
    irony_negative_policy: str = "missing"

    # emotion_spans 또는 exaggeration_spans가 누락된 Stage 2 행 처리 방식:
    # - "exclude": Bi-LSTM 학습/검증에서만 제외 (가장 안전한 기본값)
    # - "empty": 누락값을 []로 간주해 '표현 없음' 정답으로 사용
    # - "error": 기존처럼 즉시 오류 발생
    stage2_missing_span_policy: str = "exclude"

    output_dir: str = "artifacts/korean_fake_news_pipeline"

    # `klue/roberta-small`은 한국어 경량 BERT 계열 인코더입니다.
    base_model_name: str = "klue/roberta-small"

    # 긴 뉴스는 여러 조각(chunk)으로 나누고, 문서 수준에서 다시 결합합니다.
    max_length: int = 384
    bert_stride: int = 96
    index_stride: int = 0  # 어절 비율 계산 시 중복 토큰을 줄이기 위해 기본 0

    # 데이터 분할
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    group_column: Optional[str] = "group_id"  # 같은 사건/복제 기사는 같은 split에 두는 것을 권장

    # Stage 1
    stage1_batch_size: int = 4
    stage1_epochs: int = 4
    stage1_learning_rate: float = 2e-5
    stage1_weight_decay: float = 0.01
    stage1_rationale_loss_weight: float = 0.20
    stage1_total_attention_loss_weight: float = 0.10
    target_suspicious_recall: float = 0.95

    # Bi-LSTM 지수 모델
    index_batch_size: int = 8
    index_epochs: int = 6
    index_learning_rate: float = 1e-3
    index_weight_decay: float = 1e-4
    embedding_dim: int = 192
    lstm_hidden_dim: int = 192
    lstm_layers: int = 1
    index_dropout: float = 0.25
    index_score_loss_weight: float = 0.50

    # 아이러니 모델
    irony_batch_size: int = 4
    irony_epochs: int = 4
    irony_learning_rate: float = 2e-5
    irony_weight_decay: float = 0.01
    irony_rationale_loss_weight: float = 0.20

    # 공통 학습 설정
    warmup_ratio: float = 0.10
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 2
    num_workers: int = 0  # Jupyter/Windows에서 가장 안전한 기본값
    use_amp: bool = True

    # 학습 진행 표시 및 기록
    # 몇 배치마다 training_progress.csv에 기록할지 설정합니다.
    progress_log_every_n_steps: int = 10
    save_progress_log: bool = True
    show_gpu_memory: bool = True

    # 모델 내부 dropout
    bert_head_dropout: float = 0.20


KOREAN_LABELS = {
    "real": "진짜뉴스",
    "fake": "가짜뉴스",
    "satire": "풍자뉴스",
}

LABEL_ALIASES = {
    "진짜": "real",
    "진짜뉴스": "real",
    "real": "real",
    "true": "real",
    "가짜": "fake",
    "가짜뉴스": "fake",
    "fake": "fake",
    "false": "fake",
    "풍자": "satire",
    "풍자뉴스": "satire",
    "satire": "satire",
    "satirical": "satire",
}

COLUMN_ALIASES = {
    "넘버링": "id",
    "번호": "id",
    "내용": "text",
    "본문": "text",
    "라벨": "label",
    "총 attention 값": "total_attention",
    "총_attention_값": "total_attention",
    "단어/구문 attention 값": "token_attention",
    "단어_구문_attention_값": "token_attention",
    "감정자극 구문": "emotion_spans",
    "과장 구문": "exaggeration_spans",
    "아이러니 구문": "irony_spans",
    "아이러니 지수": "irony_score",
}

OPTIONAL_COLUMNS = [
    "total_attention",
    "token_attention",
    "emotion_spans",
    "exaggeration_spans",
    "irony_spans",
    "emotion_score",
    "exaggeration_score",
    "irony_score",
    "group_id",
]


def set_seed(seed: int = 42) -> None:
    """Python, NumPy, PyTorch 난수 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 완전한 결정론은 속도를 낮출 수 있어 benchmark만 끕니다.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_accelerator_cache() -> None:
    """단계별 모델을 CPU로 옮긴 뒤 가속기 캐시를 정리합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if getattr(torch, "mps", None) is not None and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def _format_duration(seconds: float) -> str:
    """초 단위 시간을 사람이 읽기 쉬운 HH:MM:SS 형태로 바꿉니다."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _accelerator_memory(device: torch.device) -> Dict[str, float]:
    """현재 CUDA 메모리 사용량을 GB 단위로 반환합니다."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_allocated_gb": 0.0,
            "gpu_reserved_gb": 0.0,
            "gpu_peak_gb": 0.0,
        }

    index = device.index if device.index is not None else torch.cuda.current_device()
    divisor = 1024 ** 3
    return {
        "gpu_allocated_gb": float(torch.cuda.memory_allocated(index) / divisor),
        "gpu_reserved_gb": float(torch.cuda.memory_reserved(index) / divisor),
        "gpu_peak_gb": float(torch.cuda.max_memory_allocated(index) / divisor),
    }


def training_progress_path(config: Config) -> Path:
    """진행 기록 CSV의 저장 위치를 반환합니다."""
    return Path(config.output_dir) / "training_progress.csv"


def initialize_training_progress(config: Config) -> Path:
    """새 학습 실행을 위해 기존 진행 기록을 초기화합니다."""
    path = training_progress_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.save_progress_log and path.exists():
        path.unlink()
    return path


def append_training_progress(
    config: Config,
    *,
    stage: str,
    status: str,
    epoch: int = 0,
    total_epochs: int = 0,
    step: int = 0,
    total_steps: int = 0,
    loss: float = float("nan"),
    learning_rate: float = float("nan"),
    elapsed_seconds: float = 0.0,
    eta_seconds: float = float("nan"),
    device: Optional[torch.device] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> None:
    """학습 진행 상태 한 줄을 CSV 파일에 추가합니다."""
    if not config.save_progress_log:
        return

    memory = _accelerator_memory(device or torch.device("cpu"))
    percent = 100.0 * step / total_steps if total_steps > 0 else float("nan")

    safe_metrics: Dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, bool, int, float)) or value is None:
            safe_metrics[str(key)] = value

    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "stage": stage,
        "status": status,
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "step": int(step),
        "total_steps": int(total_steps),
        "percent": percent,
        "loss": float(loss) if math.isfinite(float(loss)) else np.nan,
        "learning_rate": (
            float(learning_rate) if math.isfinite(float(learning_rate)) else np.nan
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "eta_seconds": (
            float(eta_seconds) if math.isfinite(float(eta_seconds)) else np.nan
        ),
        **memory,
        "metrics_json": json.dumps(
            safe_metrics,
            ensure_ascii=False,
            allow_nan=True,
        ),
    }

    path = training_progress_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def load_training_progress(output_dir: str | Path) -> pd.DataFrame:
    """저장된 training_progress.csv를 데이터프레임으로 읽습니다."""
    path = Path(output_dir) / "training_progress.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"진행 기록 파일이 아직 없습니다: {path.resolve()}"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def _update_training_progress(
    progress: tqdm,
    *,
    config: Config,
    stage: str,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    running_loss: float,
    optimizer: torch.optim.Optimizer,
    epoch_started_at: float,
    device: torch.device,
    extra_losses: Optional[Mapping[str, float]] = None,
) -> None:
    """tqdm 표시와 CSV 진행 기록을 동시에 갱신합니다."""
    elapsed = max(0.0, time.time() - epoch_started_at)
    average_loss = running_loss / max(1, step)
    eta = elapsed / max(1, step) * max(0, total_steps - step)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    memory = _accelerator_memory(device)

    postfix: Dict[str, str] = {
        "loss": f"{average_loss:.4f}",
        "lr": f"{learning_rate:.2e}",
        "ETA": _format_duration(eta),
    }
    if config.show_gpu_memory and device.type == "cuda":
        postfix["GPU"] = f"{memory['gpu_allocated_gb']:.2f}GB"

    progress.set_postfix(postfix)

    log_every = max(1, int(config.progress_log_every_n_steps))
    should_log = step == 1 or step == total_steps or step % log_every == 0
    if should_log:
        append_training_progress(
            config,
            stage=stage,
            status="training",
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            total_steps=total_steps,
            loss=average_loss,
            learning_rate=learning_rate,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            device=device,
            metrics=extra_losses,
        )


# %% [markdown]
# ## 2. 데이터 형식과 파싱
#
# 권장 열:
#
# - `id`: 고유 기사 ID
# - `text`: 기사 제목과 본문
# - `label`: `진짜뉴스`, `가짜뉴스`, `풍자뉴스`
# - `total_attention`: 0~1. 기사 전체 근거 강도/주석 신뢰도
# - `token_attention`: 일반 판별 근거 구간 JSON
# - `emotion_spans`: 감정 자극 단어/구문 JSON
# - `exaggeration_spans`: 과장 표현 단어/구문 JSON
# - `irony_spans`: 아이러니 근거 구간 JSON
# - `irony_score`: 0~1 연속값
# - `group_id`: 같은 사건, 원문 복제, 재작성 기사 묶음 ID(권장)
#
# 구간 JSON 예시:
# `[{
#   "phrase": "충격적인 진실",
#   "score": 0.95
# }]`
# 또는 정확한 문자 위치를 사용할 수 있습니다.
# `[{"start": 15, "end": 22, "score": 1.0}]`

# %%
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def safe_float(value: Any, default: float = float("nan")) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_label(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(
            f"알 수 없는 라벨: {value!r}. "
            "허용 예: 진짜뉴스, 가짜뉴스, 풍자뉴스"
        )
    return LABEL_ALIASES[key]


def _parse_json_or_literal(value: Any) -> Any:
    """JSON 문자열을 우선 파싱하고, 실패하면 Python literal 형식을 허용합니다."""
    if isinstance(value, (list, dict)):
        return value
    if _is_missing(value):
        return None

    text = str(value).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"구간 주석 JSON을 파싱할 수 없습니다: {text[:120]!r}") from exc


def parse_span_annotations(value: Any, text: str) -> Tuple[List[Dict[str, float]], bool]:
    """
    문자 구간 주석을 표준 형식으로 바꿉니다.

    반환값:
        spans: [{"start": int, "end": int, "score": float}, ...]
        available: 해당 열이 실제로 주석되었는지 여부

    `[]`는 "검토했지만 해당 표현 없음"으로 간주하고 available=True입니다.
    빈 문자열/NaN은 "주석 없음"으로 간주합니다.
    """
    if _is_missing(value):
        return [], False

    raw = _parse_json_or_literal(value)
    if raw is None:
        return [], False
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("구간 주석은 JSON 리스트 또는 딕셔너리여야 합니다.")

    spans: List[Dict[str, float]] = []

    for item in raw:
        if isinstance(item, str):
            item = {"phrase": item, "score": 1.0}
        if not isinstance(item, Mapping):
            continue

        score = _clip01(safe_float(item.get("score", item.get("attention", 1.0)), 1.0))
        start = item.get("start")
        end = item.get("end")

        # 정확한 문자 위치가 있으면 가장 우선합니다.
        if start is not None and end is not None:
            try:
                s, e = int(start), int(end)
            except (TypeError, ValueError):
                s, e = -1, -1
            if 0 <= s < e <= len(text):
                spans.append({"start": s, "end": e, "score": score})
                continue

        # 문자 위치가 없으면 phrase/text를 기사에서 검색합니다.
        phrase = item.get("phrase", item.get("text", item.get("token")))
        if phrase is None:
            continue
        phrase = str(phrase)
        if not phrase:
            continue

        # 같은 표현이 여러 번 등장하면 모두 주석합니다.
        for match in re.finditer(re.escape(phrase), text):
            spans.append(
                {
                    "start": int(match.start()),
                    "end": int(match.end()),
                    "score": score,
                }
            )

    # 중복 구간은 동일 start/end 기준 최대 점수만 유지합니다.
    merged: Dict[Tuple[int, int], float] = {}
    for span in spans:
        key = (int(span["start"]), int(span["end"]))
        merged[key] = max(merged.get(key, 0.0), float(span["score"]))

    normalized = [
        {"start": s, "end": e, "score": score}
        for (s, e), score in sorted(merged.items())
    ]
    return normalized, True


def whitespace_word_spans(text: str) -> List[Tuple[int, int, str]]:
    """한국어 뉴스의 지수 분모로 사용할 공백 기준 어절 위치를 반환합니다."""
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"\S+", text)]


def stage2_span_annotation_mask(dataframe: pd.DataFrame) -> pd.Series:
    """
    emotion_spans와 exaggeration_spans가 모두 명시된 행을 반환합니다.

    문자열 "[]"는 검토 결과 표현이 없다는 뜻이므로 유효한 주석입니다.
    빈 문자열, None, NaN은 미주석으로 처리합니다.
    """
    emotion_available = ~dataframe["emotion_spans"].map(_is_missing)
    exaggeration_available = ~dataframe["exaggeration_spans"].map(_is_missing)
    return emotion_available & exaggeration_available


def prepare_index_training_frames(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2 누락 주석 정책에 따라 Bi-LSTM 학습 프레임을 준비합니다."""
    policy = str(
        getattr(config, "stage2_missing_span_policy", "exclude")
    ).strip().lower()

    allowed = {"exclude", "empty", "error"}
    if policy not in allowed:
        raise ValueError(
            "stage2_missing_span_policy는 "
            f"{sorted(allowed)} 중 하나여야 합니다: {policy!r}"
        )

    train_frame = train_frame.copy()
    val_frame = val_frame.copy()

    train_complete = stage2_span_annotation_mask(train_frame)
    val_complete = stage2_span_annotation_mask(val_frame)

    missing_train = int((~train_complete).sum())
    missing_val = int((~val_complete).sum())

    if missing_train or missing_val:
        print(
            "Stage 2 감정·과장 주석 누락:",
            f"train={missing_train}, validation={missing_val}",
        )

        report_parts = []
        if missing_train:
            report = train_frame.loc[
                ~train_complete,
                ["id", "row_key", "label", "emotion_spans", "exaggeration_spans"],
            ].copy()
            report.insert(0, "split", "train")
            report_parts.append(report)

        if missing_val:
            report = val_frame.loc[
                ~val_complete,
                ["id", "row_key", "label", "emotion_spans", "exaggeration_spans"],
            ].copy()
            report.insert(0, "split", "validation")
            report_parts.append(report)

        report_dir = Path(config.output_dir) / "data_quality"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "missing_stage2_span_rows.csv"
        pd.concat(report_parts, ignore_index=True).to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )
        print("누락 주석 보고서:", report_path.resolve())

    if policy == "error":
        if missing_train or missing_val:
            raise ValueError(
                "Stage 2 학습 행에 emotion_spans 또는 exaggeration_spans가 "
                "누락되어 있습니다. data_quality 보고서를 확인하세요."
            )
        return train_frame, val_frame

    if policy == "empty":
        for frame in (train_frame, val_frame):
            for column in ("emotion_spans", "exaggeration_spans"):
                missing = frame[column].map(_is_missing)
                frame.loc[missing, column] = "[]"

        warnings.warn(
            "누락된 emotion_spans/exaggeration_spans를 []로 처리했습니다. "
            "누락이 실제로 '표현 없음'을 뜻할 때만 사용하세요."
        )
        return train_frame, val_frame

    # 가장 안전한 기본값: 주석이 완성된 행만 Bi-LSTM 학습·검증에 사용
    filtered_train = train_frame.loc[train_complete].copy()
    filtered_val = val_frame.loc[val_complete].copy()

    print(
        "Bi-LSTM에 사용할 주석 완료 데이터:",
        f"train={len(filtered_train)}/{len(train_frame)},",
        f"validation={len(filtered_val)}/{len(val_frame)}",
    )

    if len(filtered_train) == 0:
        raise ValueError(
            "emotion_spans와 exaggeration_spans가 모두 주석된 "
            "Stage 2 train 데이터가 없습니다."
        )

    if len(filtered_val) == 0:
        raise ValueError(
            "emotion_spans와 exaggeration_spans가 모두 주석된 "
            "Stage 2 validation 데이터가 없습니다."
        )

    return filtered_train, filtered_val


def span_annotations_to_word_ratio(
    text: str,
    spans: Sequence[Mapping[str, float]],
    score_threshold: float = 0.5,
) -> float:
    """주석된 어절 수 / 전체 어절 수를 계산합니다."""
    words = whitespace_word_spans(text)
    if not words:
        return 0.0

    marked = 0
    for ws, we, _ in words:
        is_marked = any(
            float(span.get("score", 1.0)) >= score_threshold
            and max(ws, int(span["start"])) < min(we, int(span["end"]))
            for span in spans
        )
        marked += int(is_marked)
    return marked / len(words)


def _read_csv_with_encoding(config: Config, path: Path) -> pd.DataFrame:
    """설정된 인코딩 또는 한국어 CSV의 대표 인코딩 후보로 파일을 읽습니다."""
    configured_encoding = str(getattr(config, "encoding", "auto") or "auto").strip()

    if configured_encoding.lower() != "auto":
        return pd.read_csv(
            path,
            sep=config.delimiter,
            encoding=configured_encoding,
        )

    attempted = []
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(
                path,
                sep=config.delimiter,
                encoding=encoding,
            )
            print(f"데이터 파일 인코딩: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            attempted.append(f"{encoding}: {exc}")

    details = "\n".join(attempted)
    raise UnicodeError(
        "CSV 파일을 지원 인코딩으로 읽지 못했습니다. "
        "Config의 encoding에 실제 인코딩을 직접 지정하세요.\n"
        f"시도 결과:\n{details}"
    )


def load_news_dataframe(config: Config) -> pd.DataFrame:
    """데이터 파일을 읽고 열 이름과 라벨을 정규화합니다."""
    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾지 못했습니다: {path.resolve()}\n"
            "제공된 dataset_template.csv 형식을 참고해 경로를 수정하세요."
        )

    df = _read_csv_with_encoding(config, path)
    df = df.rename(columns={c: COLUMN_ALIASES.get(c.strip(), c.strip()) for c in df.columns})

    required = {"id", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["label_norm"] = df["label"].map(normalize_label)
    df["row_key"] = np.arange(len(df), dtype=np.int64)

    # ID 중복은 결과 병합을 어렵게 하므로 경고하고 row_key를 내부 키로 사용합니다.
    if df["id"].duplicated().any():
        warnings.warn("id 열에 중복이 있습니다. 내부 병합에는 row_key를 사용합니다.")

    # total_attention, emotion_score, exaggeration_score는 반드시 0~1입니다.
    for col in ["total_attention", "emotion_score", "exaggeration_score"]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad = numeric.notna() & ~numeric.between(0.0, 1.0)
        if bad.any():
            bad_values = numeric.loc[bad].head(10).tolist()
            raise ValueError(
                f"{col} 열에는 0~1 값만 허용됩니다. "
                f"잘못된 행 수: {int(bad.sum())}, 예시: {bad_values}"
            )
        df[col] = numeric

    # irony_score의 음수값은 설정에 따라 처리합니다.
    irony = pd.to_numeric(df["irony_score"], errors="coerce")
    policy = str(
        getattr(config, "irony_negative_policy", "missing")
    ).strip().lower()

    allowed_policies = {
        "missing",
        "zero",
        "rescale_minus1_1",
        "error",
    }
    if policy not in allowed_policies:
        raise ValueError(
            "irony_negative_policy는 "
            f"{sorted(allowed_policies)} 중 하나여야 합니다: {policy!r}"
        )

    if policy == "rescale_minus1_1":
        invalid = irony.notna() & ~irony.between(-1.0, 1.0)
        if invalid.any():
            raise ValueError(
                "rescale_minus1_1 정책에서는 irony_score가 -1~1 범위여야 합니다. "
                f"잘못된 행 수: {int(invalid.sum())}"
            )
        irony = (irony + 1.0) / 2.0

    else:
        negative = irony.notna() & irony.lt(0.0)

        if negative.any():
            if policy == "error":
                raise ValueError(
                    "irony_score에 음수가 있습니다. "
                    f"음수 행 수: {int(negative.sum())}"
                )
            if policy == "zero":
                irony.loc[negative] = 0.0
                warnings.warn(
                    f"음수 irony_score {int(negative.sum())}개를 0.0으로 변환했습니다."
                )
            elif policy == "missing":
                irony.loc[negative] = np.nan
                warnings.warn(
                    f"음수 irony_score {int(negative.sum())}개를 미주석값으로 처리했습니다. "
                    "해당 행은 아이러니 모델 학습에서 제외됩니다."
                )

        above_one = irony.notna() & irony.gt(1.0)
        if above_one.any():
            bad_values = irony.loc[above_one].head(10).tolist()
            raise ValueError(
                "irony_score의 양수값은 1을 넘을 수 없습니다. "
                f"잘못된 행 수: {int(above_one.sum())}, 예시: {bad_values}"
            )

    df["irony_score"] = irony

    return df


def _can_stratify(series: pd.Series) -> bool:
    counts = series.value_counts()
    return len(counts) >= 2 and bool((counts >= 2).all())


def split_dataframe(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    train/validation/test 분할.

    group_id가 있으면 같은 사건/복제 기사 그룹이 서로 다른 split으로 새지 않도록
    GroupShuffleSplit을 사용합니다. group_id가 없으면 라벨 층화 분할을 사용합니다.
    """
    group_col = config.group_column
    use_groups = (
        group_col is not None
        and group_col in df.columns
        and df[group_col].notna().any()
        and df.loc[df[group_col].notna(), group_col].nunique() >= 3
    )

    if use_groups:
        groups = df[group_col].fillna(df["row_key"].map(lambda x: f"__single_{x}"))
        first = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_size,
            random_state=config.random_seed,
        )
        train_val_idx, test_idx = next(first.split(df, groups=groups))
        train_val = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()

        relative_val = config.val_size / (1.0 - config.test_size)
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val,
            random_state=config.random_seed + 1,
        )
        tv_groups = train_val[group_col].fillna(
            train_val["row_key"].map(lambda x: f"__single_{x}")
        )
        train_idx, val_idx = next(second.split(train_val, groups=tv_groups))
        train_df = train_val.iloc[train_idx].copy()
        val_df = train_val.iloc[val_idx].copy()
    else:
        stratify = df["label_norm"] if _can_stratify(df["label_norm"]) else None
        train_val, test_df = train_test_split(
            df,
            test_size=config.test_size,
            random_state=config.random_seed,
            stratify=stratify,
        )
        relative_val = config.val_size / (1.0 - config.test_size)
        stratify_tv = train_val["label_norm"] if _can_stratify(train_val["label_norm"]) else None
        train_df, val_df = train_test_split(
            train_val,
            test_size=relative_val,
            random_state=config.random_seed + 1,
            stratify=stratify_tv,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "train": train_df["label_norm"].value_counts(),
            "validation": val_df["label_norm"].value_counts(),
            "test": test_df["label_norm"].value_counts(),
        }
    ).fillna(0).astype(int)
    print(summary)


# %% [markdown]
# ## 3. 토큰화·문자 구간 정렬

# %%
def tokenize_document(
    tokenizer,
    text: str,
    max_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """긴 문서를 겹치는 BERT 조각으로 토큰화합니다."""
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        add_special_tokens=True,
    )

    chunks: List[Dict[str, Any]] = []
    n_chunks = len(encoded["input_ids"])
    for i in range(n_chunks):
        model_inputs = {
            key: encoded[key][i]
            for key in tokenizer.model_input_names
            if key in encoded
        }
        offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"][i]]
        chunks.append({"model_inputs": model_inputs, "offset_mapping": offsets})
    return chunks


def char_spans_to_token_scores(
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Mapping[str, float]],
) -> List[float]:
    """문자 단위 근거 구간을 토큰 단위 0~1 점수로 정렬합니다."""
    scores: List[float] = []
    for start, end in offsets:
        if end <= start:  # [CLS], [SEP], [PAD] 같은 특수 토큰
            scores.append(0.0)
            continue

        token_score = 0.0
        for span in spans:
            s, e = int(span["start"]), int(span["end"])
            if max(start, s) < min(end, e):
                token_score = max(token_score, float(span.get("score", 1.0)))
        scores.append(_clip01(token_score))
    return scores


# %% [markdown]
# ## 4. Dataset과 Collator

# %%
class Stage1DocumentDataset(Dataset):
    """진짜(0) 대 의심=가짜+풍자(1) 문서 데이터셋."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index in self._cache:
            return self._cache[index]

        row = self.df.iloc[index]
        text = str(row["text"])
        spans, rationale_available = parse_span_annotations(row["token_attention"], text)
        chunks = tokenize_document(self.tokenizer, text, self.max_length, self.stride)

        for chunk in chunks:
            chunk["rationale_target"] = char_spans_to_token_scores(
                chunk["offset_mapping"], spans
            )
            chunk["rationale_available"] = rationale_available

        item = {
            "id": row["id"],
            "row_key": int(row["row_key"]),
            "text": text,
            "chunks": chunks,
            "label": float(row["label_norm"] in {"fake", "satire"}),
            "label_norm": row["label_norm"],
            "total_attention": safe_float(row["total_attention"]),
        }
        self._cache[index] = item
        return item


class IndexDocumentDataset(Dataset):
    """감정자극·과장 표현 token supervision을 위한 Bi-LSTM 데이터셋."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        stride: int = 0,
        require_targets: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.require_targets = require_targets
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index in self._cache:
            return self._cache[index]

        row = self.df.iloc[index]
        text = str(row["text"])
        emotion_spans, emotion_available = parse_span_annotations(row["emotion_spans"], text)
        exaggeration_spans, exaggeration_available = parse_span_annotations(
            row["exaggeration_spans"], text
        )

        if self.require_targets and row["label_norm"] in {"fake", "satire"}:
            if not emotion_available or not exaggeration_available:
                raise ValueError(
                    f"row_key={row['row_key']}의 Stage 2 학습 주석이 없습니다. "
                    "emotion_spans와 exaggeration_spans에 최소한 []를 넣어 주세요."
                )

        emotion_score = safe_float(row.get("emotion_score", np.nan))
        exaggeration_score = safe_float(row.get("exaggeration_score", np.nan))
        if math.isnan(emotion_score) and emotion_available:
            emotion_score = span_annotations_to_word_ratio(text, emotion_spans)
        if math.isnan(exaggeration_score) and exaggeration_available:
            exaggeration_score = span_annotations_to_word_ratio(text, exaggeration_spans)

        chunks = tokenize_document(self.tokenizer, text, self.max_length, self.stride)
        for chunk in chunks:
            chunk["emotion_target"] = char_spans_to_token_scores(
                chunk["offset_mapping"], emotion_spans
            )
            chunk["exaggeration_target"] = char_spans_to_token_scores(
                chunk["offset_mapping"], exaggeration_spans
            )
            chunk["emotion_available"] = emotion_available
            chunk["exaggeration_available"] = exaggeration_available

        label_norm = row["label_norm"]
        stage2_label = 1 if label_norm == "satire" else 0 if label_norm == "fake" else -1
        item = {
            "id": row["id"],
            "row_key": int(row["row_key"]),
            "text": text,
            "chunks": chunks,
            "emotion_score": emotion_score,
            "exaggeration_score": exaggeration_score,
            "stage2_label": stage2_label,
            "label_norm": label_norm,
        }
        self._cache[index] = item
        return item

    def token_class_counts(self) -> Dict[str, Tuple[float, float]]:
        """희소한 양성 토큰의 pos_weight 계산용 통계."""
        counts = {
            "emotion": [0.0, 0.0],  # positive, negative
            "exaggeration": [0.0, 0.0],
        }
        for i in tqdm(range(len(self)), desc="토큰 클래스 비율 계산", leave=False):
            item = self[i]
            for chunk in item["chunks"]:
                offsets = chunk["offset_mapping"]
                content = np.array([e > s for s, e in offsets], dtype=bool)
                if chunk["emotion_available"]:
                    y = np.asarray(chunk["emotion_target"])[content]
                    counts["emotion"][0] += float((y >= 0.5).sum())
                    counts["emotion"][1] += float((y < 0.5).sum())
                if chunk["exaggeration_available"]:
                    y = np.asarray(chunk["exaggeration_target"])[content]
                    counts["exaggeration"][0] += float((y >= 0.5).sum())
                    counts["exaggeration"][1] += float((y < 0.5).sum())
        return {key: (value[0], value[1]) for key, value in counts.items()}


class IronyDocumentDataset(Dataset):
    """아이러니 지수(0~1) 회귀와 아이러니 근거 attention 지도 데이터셋."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
        stride: int,
        require_targets: bool = True,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.require_targets = require_targets
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index in self._cache:
            return self._cache[index]

        row = self.df.iloc[index]
        text = str(row["text"])
        spans, rationale_available = parse_span_annotations(row["irony_spans"], text)
        irony_score = safe_float(row["irony_score"])

        if self.require_targets and row["label_norm"] in {"fake", "satire"}:
            if math.isnan(irony_score):
                raise ValueError(
                    f"row_key={row['row_key']}의 irony_score가 없습니다. "
                    "가짜/풍자 학습 행에는 0~1 값을 넣어 주세요."
                )

        chunks = tokenize_document(self.tokenizer, text, self.max_length, self.stride)
        for chunk in chunks:
            chunk["rationale_target"] = char_spans_to_token_scores(
                chunk["offset_mapping"], spans
            )
            chunk["rationale_available"] = rationale_available

        label_norm = row["label_norm"]
        stage2_label = 1 if label_norm == "satire" else 0 if label_norm == "fake" else -1
        item = {
            "id": row["id"],
            "row_key": int(row["row_key"]),
            "text": text,
            "chunks": chunks,
            "irony_score": irony_score,
            "stage2_label": stage2_label,
            "label_norm": label_norm,
        }
        self._cache[index] = item
        return item


def _flatten_chunks(items: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    flat: List[Dict[str, Any]] = []
    chunk_to_doc: List[int] = []
    for doc_index, item in enumerate(items):
        for chunk in item["chunks"]:
            flat.append(chunk)
            chunk_to_doc.append(doc_index)
    return flat, torch.tensor(chunk_to_doc, dtype=torch.long)


def _pad_model_inputs(tokenizer, flat_chunks: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    features = [chunk["model_inputs"] for chunk in flat_chunks]
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def _pad_offsets(flat_chunks: Sequence[Dict[str, Any]], max_len: int) -> torch.Tensor:
    output = torch.full((len(flat_chunks), max_len, 2), -1, dtype=torch.long)
    for i, chunk in enumerate(flat_chunks):
        offsets = torch.tensor(chunk["offset_mapping"], dtype=torch.long)
        output[i, : len(offsets)] = offsets
    return output


def _pad_float_field(
    flat_chunks: Sequence[Dict[str, Any]],
    field: str,
    max_len: int,
) -> torch.Tensor:
    output = torch.zeros((len(flat_chunks), max_len), dtype=torch.float32)
    for i, chunk in enumerate(flat_chunks):
        values = torch.tensor(chunk[field], dtype=torch.float32)
        output[i, : len(values)] = values
    return output


class Stage1Collator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        flat, chunk_to_doc = _flatten_chunks(items)
        batch = _pad_model_inputs(self.tokenizer, flat)
        max_len = batch["input_ids"].shape[1]
        offsets = _pad_offsets(flat, max_len)

        batch.update(
            {
                "offset_mapping": offsets,
                "content_mask": offsets[..., 1] > offsets[..., 0],
                "rationale_targets": _pad_float_field(flat, "rationale_target", max_len),
                "rationale_available": torch.tensor(
                    [bool(c["rationale_available"]) for c in flat], dtype=torch.bool
                ),
                "chunk_to_doc": chunk_to_doc,
                "num_docs": len(items),
                "labels": torch.tensor([item["label"] for item in items], dtype=torch.float32),
                "total_attention": torch.tensor(
                    [item["total_attention"] for item in items], dtype=torch.float32
                ),
                "ids": [item["id"] for item in items],
                "row_keys": [item["row_key"] for item in items],
                "texts": [item["text"] for item in items],
                "label_norms": [item["label_norm"] for item in items],
            }
        )
        return batch


class IndexCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        flat, chunk_to_doc = _flatten_chunks(items)
        batch = _pad_model_inputs(self.tokenizer, flat)
        max_len = batch["input_ids"].shape[1]
        offsets = _pad_offsets(flat, max_len)

        batch.update(
            {
                "offset_mapping": offsets,
                "content_mask": offsets[..., 1] > offsets[..., 0],
                "emotion_targets": _pad_float_field(flat, "emotion_target", max_len),
                "exaggeration_targets": _pad_float_field(
                    flat, "exaggeration_target", max_len
                ),
                "emotion_available": torch.tensor(
                    [bool(c["emotion_available"]) for c in flat], dtype=torch.bool
                ),
                "exaggeration_available": torch.tensor(
                    [bool(c["exaggeration_available"]) for c in flat], dtype=torch.bool
                ),
                "chunk_to_doc": chunk_to_doc,
                "num_docs": len(items),
                "emotion_scores": torch.tensor(
                    [item["emotion_score"] for item in items], dtype=torch.float32
                ),
                "exaggeration_scores": torch.tensor(
                    [item["exaggeration_score"] for item in items], dtype=torch.float32
                ),
                "stage2_labels": torch.tensor(
                    [item["stage2_label"] for item in items], dtype=torch.long
                ),
                "ids": [item["id"] for item in items],
                "row_keys": [item["row_key"] for item in items],
                "texts": [item["text"] for item in items],
                "label_norms": [item["label_norm"] for item in items],
            }
        )
        return batch


class IronyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        flat, chunk_to_doc = _flatten_chunks(items)
        batch = _pad_model_inputs(self.tokenizer, flat)
        max_len = batch["input_ids"].shape[1]
        offsets = _pad_offsets(flat, max_len)

        batch.update(
            {
                "offset_mapping": offsets,
                "content_mask": offsets[..., 1] > offsets[..., 0],
                "rationale_targets": _pad_float_field(flat, "rationale_target", max_len),
                "rationale_available": torch.tensor(
                    [bool(c["rationale_available"]) for c in flat], dtype=torch.bool
                ),
                "chunk_to_doc": chunk_to_doc,
                "num_docs": len(items),
                "irony_scores": torch.tensor(
                    [item["irony_score"] for item in items], dtype=torch.float32
                ),
                "stage2_labels": torch.tensor(
                    [item["stage2_label"] for item in items], dtype=torch.long
                ),
                "ids": [item["id"] for item in items],
                "row_keys": [item["row_key"] for item in items],
                "texts": [item["text"] for item in items],
                "label_norms": [item["label_norm"] for item in items],
            }
        )
        return batch


# %% [markdown]
# ## 5. 모델 정의

# %%
def _build_encoder(
    model_name: Optional[str] = None,
    encoder_config=None,
    pretrained: bool = True,
):
    if pretrained:
        if not model_name:
            raise ValueError("pretrained=True일 때 model_name이 필요합니다.")
        return AutoModel.from_pretrained(model_name)
    if encoder_config is None:
        raise ValueError("pretrained=False일 때 encoder_config가 필요합니다.")
    return AutoModel.from_config(encoder_config)


class HierarchicalRationaleEncoder(nn.Module):
    """
    토큰 attention pooling -> chunk attention pooling의 2단계 문서 인코더.

    토큰 attention은 LLM이 표시한 근거 구간을 보조 정답으로 학습할 수 있습니다.
    긴 뉴스는 여러 chunk를 만든 뒤 chunk attention으로 문서 표현을 합칩니다.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        encoder_config=None,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = _build_encoder(model_name, encoder_config, pretrained)
        hidden_size = int(self.encoder.config.hidden_size)
        self.token_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.chunk_attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.accepts_token_type_ids = (
            "token_type_ids" in inspect.signature(self.encoder.forward).parameters
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        chunk_to_doc: torch.Tensor,
        num_docs: int,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids

        hidden = self.encoder(**encoder_kwargs).last_hidden_state

        # 특수 토큰을 제외하고 attention을 계산합니다.
        token_mask = content_mask.bool()
        no_content = ~token_mask.any(dim=1)
        if no_content.any():
            token_mask = token_mask.clone()
            token_mask[no_content] = attention_mask[no_content].bool()

        token_logits = self.token_attention(hidden).squeeze(-1)
        token_logits = token_logits.masked_fill(~token_mask, -1e4)
        token_weights = torch.softmax(token_logits, dim=-1)
        chunk_repr = torch.bmm(token_weights.unsqueeze(1), hidden).squeeze(1)
        chunk_repr = self.dropout(chunk_repr)

        # 배치 안에서 문서별로 가변 개수의 chunk를 attention pooling합니다.
        #
        # AMP에서는 gate_logits와 chunk_repr가 float16이 될 수 있지만,
        # softmax는 수치 안정성을 위해 float32로 계산하는 것이 안전합니다.
        # attention 확률은 float32로 저장하고, pooling할 때만
        # chunk_repr의 dtype으로 변환합니다.
        gate_logits = self.chunk_attention(chunk_repr).squeeze(-1)
        doc_representations: List[torch.Tensor] = []
        chunk_weights = torch.zeros(
            gate_logits.shape,
            dtype=torch.float32,
            device=gate_logits.device,
        )

        for doc_index in range(num_docs):
            mask = chunk_to_doc == doc_index
            if not mask.any():
                raise RuntimeError(f"문서 {doc_index}에 대응하는 chunk가 없습니다.")

            weights_fp32 = torch.softmax(
                gate_logits[mask].float(),
                dim=0,
            )
            chunk_weights[mask] = weights_fp32

            weights_for_pool = weights_fp32.to(dtype=chunk_repr.dtype)
            doc_representations.append(
                (
                    weights_for_pool.unsqueeze(-1)
                    * chunk_repr[mask]
                ).sum(dim=0)
            )

        doc_repr = torch.stack(doc_representations, dim=0)
        return {
            "doc_repr": doc_repr,
            "token_attention": token_weights,
            "chunk_attention": chunk_weights,
        }


class Stage1SuspicionModel(nn.Module):
    """진짜 대 의심(가짜+풍자) 이진 확률 모델."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        encoder_config=None,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.document_encoder = HierarchicalRationaleEncoder(
            model_name=model_name,
            encoder_config=encoder_config,
            pretrained=pretrained,
            dropout=dropout,
        )
        hidden = self.document_encoder.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.total_attention_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        encoded = self.document_encoder(**kwargs)
        encoded["doc_logits"] = self.classifier(encoded["doc_repr"]).squeeze(-1)
        encoded["total_attention_logits"] = self.total_attention_head(
            encoded["doc_repr"]
        ).squeeze(-1)
        return encoded


class IronyRegressor(nn.Module):
    """아이러니 지수 0~1을 예측하는 경량 한국어 BERT 모델."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        encoder_config=None,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.document_encoder = HierarchicalRationaleEncoder(
            model_name=model_name,
            encoder_config=encoder_config,
            pretrained=pretrained,
            dropout=dropout,
        )
        hidden = self.document_encoder.hidden_size
        self.score_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        encoded = self.document_encoder(**kwargs)
        encoded["score_logits"] = self.score_head(encoded["doc_repr"]).squeeze(-1)
        return encoded


class DualIndexBiLSTM(nn.Module):
    """하나의 Bi-LSTM trunk에서 감정자극·과장 토큰을 각각 예측합니다."""

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        embedding_dim: int = 192,
        hidden_dim: int = 192,
        num_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden_dim * 2, 1)
        self.exaggeration_head = nn.Linear(hidden_dim * 2, 1)

        self.model_config = {
            "vocab_size": vocab_size,
            "padding_idx": padding_idx,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    @staticmethod
    def _document_soft_ratio(
        token_probabilities: torch.Tensor,
        content_mask: torch.Tensor,
        chunk_to_doc: torch.Tensor,
        num_docs: int,
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for doc_index in range(num_docs):
            doc_chunks = chunk_to_doc == doc_index
            valid = doc_chunks.unsqueeze(-1) & content_mask.bool()
            values = token_probabilities[valid]
            if values.numel() == 0:
                outputs.append(token_probabilities.new_tensor(0.0))
            else:
                outputs.append(values.mean())
        return torch.stack(outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        chunk_to_doc: torch.Tensor,
        num_docs: int,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp_min(1).to("cpu")
        packed = pack_padded_sequence(
            embedded,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.shape[1],
        )
        output = self.dropout(output)

        emotion_logits = self.emotion_head(output).squeeze(-1)
        exaggeration_logits = self.exaggeration_head(output).squeeze(-1)
        emotion_prob = torch.sigmoid(emotion_logits)
        exaggeration_prob = torch.sigmoid(exaggeration_logits)

        return {
            "emotion_logits": emotion_logits,
            "exaggeration_logits": exaggeration_logits,
            # 학습 중 문서 지수 회귀에 사용하는 미분 가능한 근사값입니다.
            "emotion_soft_score": self._document_soft_ratio(
                emotion_prob, content_mask, chunk_to_doc, num_docs
            ),
            "exaggeration_soft_score": self._document_soft_ratio(
                exaggeration_prob, content_mask, chunk_to_doc, num_docs
            ),
        }


# %% [markdown]
# ## 6. 손실 함수

# %%
def _rationale_cross_entropy(
    predicted_attention: torch.Tensor,
    targets: torch.Tensor,
    available: torch.Tensor,
    chunk_to_doc: torch.Tensor,
    document_confidence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """정규화된 rationale 분포와 모델 attention 분포의 교차 엔트로피."""
    target_sum = targets.sum(dim=1)
    valid = available.bool() & (target_sum > 0)
    if not valid.any():
        return predicted_attention.new_tensor(0.0)

    target_distribution = targets[valid] / target_sum[valid].unsqueeze(-1).clamp_min(1e-8)
    per_chunk = -(
        target_distribution * torch.log(predicted_attention[valid].clamp_min(1e-8))
    ).sum(dim=1)

    if document_confidence is not None:
        confidence = document_confidence[chunk_to_doc[valid]]
        confidence = torch.where(
            torch.isfinite(confidence),
            0.25 + 0.75 * confidence.clamp(0.0, 1.0),
            torch.ones_like(confidence),
        )
        per_chunk = per_chunk * confidence

    return per_chunk.mean()


def stage1_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    positive_weight: torch.Tensor,
    rationale_weight: float,
    total_attention_weight: float,
) -> Dict[str, torch.Tensor]:
    classification = F.binary_cross_entropy_with_logits(
        outputs["doc_logits"],
        batch["labels"],
        pos_weight=positive_weight,
    )
    rationale = _rationale_cross_entropy(
        outputs["token_attention"],
        batch["rationale_targets"],
        batch["rationale_available"],
        batch["chunk_to_doc"],
        batch["total_attention"],
    )

    valid_total = torch.isfinite(batch["total_attention"])
    if valid_total.any():
        predicted_total = torch.sigmoid(outputs["total_attention_logits"][valid_total])
        total_attention = F.mse_loss(predicted_total, batch["total_attention"][valid_total])
    else:
        total_attention = classification.new_tensor(0.0)

    total = classification + rationale_weight * rationale + total_attention_weight * total_attention
    return {
        "loss": total,
        "classification": classification.detach(),
        "rationale": rationale.detach(),
        "total_attention": total_attention.detach(),
    }


def _masked_token_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    content_mask: torch.Tensor,
    annotation_available: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    mask = content_mask.bool() & annotation_available.unsqueeze(-1)
    if not mask.any():
        return logits.new_tensor(0.0)

    loss_matrix = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    return loss_matrix[mask].mean()


def index_model_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    emotion_positive_weight: torch.Tensor,
    exaggeration_positive_weight: torch.Tensor,
    score_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    emotion_token = _masked_token_bce(
        outputs["emotion_logits"],
        batch["emotion_targets"],
        batch["content_mask"],
        batch["emotion_available"],
        emotion_positive_weight,
    )
    exaggeration_token = _masked_token_bce(
        outputs["exaggeration_logits"],
        batch["exaggeration_targets"],
        batch["content_mask"],
        batch["exaggeration_available"],
        exaggeration_positive_weight,
    )

    valid_emotion = torch.isfinite(batch["emotion_scores"])
    valid_exaggeration = torch.isfinite(batch["exaggeration_scores"])

    emotion_score_loss = (
        F.mse_loss(
            outputs["emotion_soft_score"][valid_emotion],
            batch["emotion_scores"][valid_emotion],
        )
        if valid_emotion.any()
        else emotion_token.new_tensor(0.0)
    )
    exaggeration_score_loss = (
        F.mse_loss(
            outputs["exaggeration_soft_score"][valid_exaggeration],
            batch["exaggeration_scores"][valid_exaggeration],
        )
        if valid_exaggeration.any()
        else emotion_token.new_tensor(0.0)
    )

    token_total = emotion_token + exaggeration_token
    score_total = emotion_score_loss + exaggeration_score_loss
    total = token_total + score_loss_weight * score_total
    return {
        "loss": total,
        "emotion_token": emotion_token.detach(),
        "exaggeration_token": exaggeration_token.detach(),
        "emotion_score": emotion_score_loss.detach(),
        "exaggeration_score": exaggeration_score_loss.detach(),
    }


def irony_model_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    rationale_weight: float,
) -> Dict[str, torch.Tensor]:
    valid = torch.isfinite(batch["irony_scores"])
    if not valid.any():
        raise ValueError("현재 배치에 유효한 irony_score가 없습니다.")

    # BCEWithLogitsLoss는 0~1 soft target에도 사용할 수 있습니다.
    regression = F.binary_cross_entropy_with_logits(
        outputs["score_logits"][valid], batch["irony_scores"][valid]
    )
    rationale = _rationale_cross_entropy(
        outputs["token_attention"],
        batch["rationale_targets"],
        batch["rationale_available"],
        batch["chunk_to_doc"],
        document_confidence=None,
    )
    total = regression + rationale_weight * rationale
    return {
        "loss": total,
        "regression": regression.detach(),
        "rationale": rationale.detach(),
    }


# %% [markdown]
# ## 7. 공통 학습 유틸리티

# %%
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def encoder_forward_kwargs(batch: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "content_mask": batch["content_mask"],
        "chunk_to_doc": batch["chunk_to_doc"],
        "num_docs": int(batch["num_docs"]),
    }
    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]
    return kwargs


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    active = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=active)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=active)


def make_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_ratio: float,
):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    parameter_groups = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if p.requires_grad and not any(nd in name for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if p.requires_grad and any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def safe_metric(func, y_true, y_score, default: float = float("nan")) -> float:
    try:
        return float(func(y_true, y_score))
    except ValueError:
        return default


def compact_scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """metadata가 거대해지지 않도록 스칼라 평가값만 보존합니다."""
    compact: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
            compact[key] = _json_ready(value) if "_json_ready" in globals() else (
                value.item() if isinstance(value, np.generic) else value
            )
    return compact


# %% [markdown]
# ## 8. Stage 1 학습·확률 보정·임계값 선택

# %%
@torch.no_grad()
def evaluate_stage1(
    model: Stage1SuspicionModel,
    loader: DataLoader,
    device: torch.device,
    positive_weight: torch.Tensor,
    config: Config,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    logits: List[float] = []
    labels: List[int] = []
    row_keys: List[int] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        loss_dict = stage1_loss(
            outputs,
            batch,
            positive_weight,
            config.stage1_rationale_loss_weight,
            config.stage1_total_attention_loss_weight,
        )
        losses.append(float(loss_dict["loss"].item()))
        logits.extend(outputs["doc_logits"].detach().cpu().tolist())
        labels.extend(batch["labels"].long().cpu().tolist())
        row_keys.extend(raw_batch["row_keys"])

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    binary = (probabilities >= 0.5).astype(int)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy@0.5": float(accuracy_score(labels, binary)),
        "f1@0.5": float(f1_score(labels, binary, zero_division=0)),
        "recall@0.5": float(recall_score(labels, binary, zero_division=0)),
        "roc_auc": safe_metric(roc_auc_score, labels, probabilities),
        "average_precision": safe_metric(average_precision_score, labels, probabilities),
        "logits": np.asarray(logits, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "row_keys": np.asarray(row_keys, dtype=np.int64),
    }
    return metrics


def train_stage1(
    model: Stage1SuspicionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    device: torch.device,
    config: Config,
) -> Tuple[Stage1SuspicionModel, Dict[str, Any]]:
    positives = int(train_df["label_norm"].isin(["fake", "satire"]).sum())
    negatives = len(train_df) - positives
    positive_weight = torch.tensor(
        max(0.25, min(20.0, negatives / max(1, positives))),
        dtype=torch.float32,
        device=device,
    )

    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        config.stage1_learning_rate,
        config.stage1_weight_decay,
        config.stage1_epochs,
        len(train_loader),
        config.warmup_ratio,
    )
    scaler = make_grad_scaler(device, config.use_amp)

    best_state = cpu_state_dict(model)
    best_score = -float("inf")
    best_metrics: Dict[str, Any] = {}
    patience = 0

    for epoch in range(1, config.stage1_epochs + 1):
        model.train()
        total_steps = len(train_loader)
        epoch_started_at = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        progress = tqdm(
            train_loader,
            desc=f"Stage 1 [{epoch}/{config.stage1_epochs}]",
            leave=True,
            dynamic_ncols=True,
        )
        running = 0.0
        running_parts = {
            "classification": 0.0,
            "rationale": 0.0,
            "total_attention": 0.0,
        }

        for step, raw_batch in enumerate(progress, start=1):
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, config.use_amp):
                outputs = model(**encoder_forward_kwargs(batch))
                loss_dict = stage1_loss(
                    outputs,
                    batch,
                    positive_weight,
                    config.stage1_rationale_loss_weight,
                    config.stage1_total_attention_loss_weight,
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            for key in running_parts:
                running_parts[key] += float(loss_dict[key].item())

            average_parts = {
                key: value / step
                for key, value in running_parts.items()
            }
            _update_training_progress(
                progress,
                config=config,
                stage="stage1",
                epoch=epoch,
                total_epochs=config.stage1_epochs,
                step=step,
                total_steps=total_steps,
                running_loss=running,
                optimizer=optimizer,
                epoch_started_at=epoch_started_at,
                device=device,
                extra_losses=average_parts,
            )

        metrics = evaluate_stage1(model, val_loader, device, positive_weight, config)
        score = metrics["average_precision"]
        if math.isnan(score):
            score = -metrics["loss"]
        print(
            f"[Stage 1] epoch={epoch} val_loss={metrics['loss']:.4f} "
            f"AP={metrics['average_precision']:.4f} AUC={metrics['roc_auc']:.4f}"
        )
        append_training_progress(
            config,
            stage="stage1",
            status="validation",
            epoch=epoch,
            total_epochs=config.stage1_epochs,
            step=len(train_loader),
            total_steps=len(train_loader),
            loss=metrics["loss"],
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.time() - epoch_started_at,
            eta_seconds=0.0,
            device=device,
            metrics=compact_scalar_metrics(metrics),
        )

        if score > best_score + 1e-6:
            best_score = score
            best_state = cpu_state_dict(model)
            best_metrics = compact_scalar_metrics(metrics)
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("Stage 1 조기 종료")
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, {"positive_weight": float(positive_weight.item()), "best": best_metrics}


class TemperatureScaler:
    """검증 세트에서 이진 분류 logit의 온도 하나를 학습합니다."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logit_tensor = torch.tensor(logits, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        log_temperature = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(0.05, 20.0)
            loss = F.binary_cross_entropy_with_logits(
                logit_tensor / temperature, label_tensor
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        values = torch.tensor(logits, dtype=torch.float32) / self.temperature
        return torch.sigmoid(values).numpy()


def choose_suspicion_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float = 0.95,
) -> Dict[str, float]:
    """의심 뉴스 재현율을 우선 보장하면서 F1이 가장 좋은 임계값을 선택합니다."""
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.02, 0.98, 193),
                probabilities,
            ]
        )
    )
    rows: List[Dict[str, float]] = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "recall": float(recall_score(labels, pred, zero_division=0)),
                "precision": float(precision_score(labels, pred, zero_division=0)),
                "f1": float(f1_score(labels, pred, zero_division=0)),
            }
        )

    eligible = [row for row in rows if row["recall"] >= target_recall]
    pool = eligible if eligible else rows
    best = max(pool, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    best["target_recall_satisfied"] = float(bool(eligible))
    return best


# %% [markdown]
# ## 9. Bi-LSTM 감정자극·과장 지수 모델 학습

# %%
@torch.no_grad()
def evaluate_index_model(
    model: DualIndexBiLSTM,
    loader: DataLoader,
    device: torch.device,
    emotion_positive_weight: torch.Tensor,
    exaggeration_positive_weight: torch.Tensor,
    config: Config,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    emotion_true: List[float] = []
    emotion_pred: List[float] = []
    exag_true: List[float] = []
    exag_pred: List[float] = []
    token_emotion_true: List[float] = []
    token_emotion_prob: List[float] = []
    token_exag_true: List[float] = []
    token_exag_prob: List[float] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        loss_dict = index_model_loss(
            outputs,
            batch,
            emotion_positive_weight,
            exaggeration_positive_weight,
            config.index_score_loss_weight,
        )
        losses.append(float(loss_dict["loss"].item()))

        valid_e_score = torch.isfinite(batch["emotion_scores"])
        valid_x_score = torch.isfinite(batch["exaggeration_scores"])
        emotion_true.extend(batch["emotion_scores"][valid_e_score].cpu().tolist())
        emotion_pred.extend(outputs["emotion_soft_score"][valid_e_score].cpu().tolist())
        exag_true.extend(batch["exaggeration_scores"][valid_x_score].cpu().tolist())
        exag_pred.extend(outputs["exaggeration_soft_score"][valid_x_score].cpu().tolist())

        e_mask = batch["content_mask"] & batch["emotion_available"].unsqueeze(-1)
        x_mask = batch["content_mask"] & batch["exaggeration_available"].unsqueeze(-1)
        token_emotion_true.extend(batch["emotion_targets"][e_mask].cpu().tolist())
        token_emotion_prob.extend(torch.sigmoid(outputs["emotion_logits"])[e_mask].cpu().tolist())
        token_exag_true.extend(batch["exaggeration_targets"][x_mask].cpu().tolist())
        token_exag_prob.extend(
            torch.sigmoid(outputs["exaggeration_logits"])[x_mask].cpu().tolist()
        )

    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "emotion_mae_soft": float(mean_absolute_error(emotion_true, emotion_pred))
        if emotion_true
        else float("nan"),
        "exaggeration_mae_soft": float(mean_absolute_error(exag_true, exag_pred))
        if exag_true
        else float("nan"),
        "emotion_token_true": np.asarray(token_emotion_true, dtype=np.float32),
        "emotion_token_prob": np.asarray(token_emotion_prob, dtype=np.float32),
        "exaggeration_token_true": np.asarray(token_exag_true, dtype=np.float32),
        "exaggeration_token_prob": np.asarray(token_exag_prob, dtype=np.float32),
    }
    return metrics


def _positive_weight_from_counts(positive: float, negative: float) -> float:
    if positive <= 0:
        warnings.warn("양성 토큰이 없습니다. pos_weight=1.0을 사용합니다.")
        return 1.0
    return float(np.clip(negative / positive, 0.5, 30.0))


def train_index_model(
    model: DualIndexBiLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_dataset: IndexDocumentDataset,
    device: torch.device,
    config: Config,
) -> Tuple[DualIndexBiLSTM, Dict[str, Any]]:
    counts = train_dataset.token_class_counts()
    e_pos, e_neg = counts["emotion"]
    x_pos, x_neg = counts["exaggeration"]
    e_weight = torch.tensor(
        _positive_weight_from_counts(e_pos, e_neg), device=device, dtype=torch.float32
    )
    x_weight = torch.tensor(
        _positive_weight_from_counts(x_pos, x_neg), device=device, dtype=torch.float32
    )

    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        config.index_learning_rate,
        config.index_weight_decay,
        config.index_epochs,
        len(train_loader),
        config.warmup_ratio,
    )
    scaler = make_grad_scaler(device, config.use_amp)
    best_state = cpu_state_dict(model)
    best_loss = float("inf")
    best_metrics: Dict[str, Any] = {}
    patience = 0

    for epoch in range(1, config.index_epochs + 1):
        model.train()
        total_steps = len(train_loader)
        epoch_started_at = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        progress = tqdm(
            train_loader,
            desc=f"Bi-LSTM [{epoch}/{config.index_epochs}]",
            leave=True,
            dynamic_ncols=True,
        )
        running = 0.0
        running_parts = {
            "emotion_token": 0.0,
            "exaggeration_token": 0.0,
            "emotion_score": 0.0,
            "exaggeration_score": 0.0,
        }

        for step, raw_batch in enumerate(progress, start=1):
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, config.use_amp):
                outputs = model(**encoder_forward_kwargs(batch))
                loss_dict = index_model_loss(
                    outputs,
                    batch,
                    e_weight,
                    x_weight,
                    config.index_score_loss_weight,
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            for key in running_parts:
                running_parts[key] += float(loss_dict[key].item())

            average_parts = {
                key: value / step
                for key, value in running_parts.items()
            }
            _update_training_progress(
                progress,
                config=config,
                stage="bilstm",
                epoch=epoch,
                total_epochs=config.index_epochs,
                step=step,
                total_steps=total_steps,
                running_loss=running,
                optimizer=optimizer,
                epoch_started_at=epoch_started_at,
                device=device,
                extra_losses=average_parts,
            )

        metrics = evaluate_index_model(
            model, val_loader, device, e_weight, x_weight, config
        )
        print(
            f"[Bi-LSTM] epoch={epoch} val_loss={metrics['loss']:.4f} "
            f"emotion_MAE={metrics['emotion_mae_soft']:.4f} "
            f"exaggeration_MAE={metrics['exaggeration_mae_soft']:.4f}"
        )
        append_training_progress(
            config,
            stage="bilstm",
            status="validation",
            epoch=epoch,
            total_epochs=config.index_epochs,
            step=len(train_loader),
            total_steps=len(train_loader),
            loss=metrics["loss"],
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.time() - epoch_started_at,
            eta_seconds=0.0,
            device=device,
            metrics=compact_scalar_metrics(metrics),
        )

        if metrics["loss"] < best_loss - 1e-6:
            best_loss = metrics["loss"]
            best_state = cpu_state_dict(model)
            best_metrics = compact_scalar_metrics(metrics)
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("Bi-LSTM 조기 종료")
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "emotion_positive_weight": float(e_weight.item()),
        "exaggeration_positive_weight": float(x_weight.item()),
        "best": best_metrics,
    }


def choose_token_threshold(targets: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    if len(targets) == 0:
        return {"threshold": 0.5, "f1": float("nan")}
    y_true = (targets >= 0.5).astype(int)
    candidates = np.linspace(0.05, 0.95, 91)
    rows = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
            }
        )
    return max(rows, key=lambda row: (row["f1"], row["recall"]))


# %% [markdown]
# ## 10. 아이러니 BERT 학습

# %%
@torch.no_grad()
def evaluate_irony_model(
    model: IronyRegressor,
    loader: DataLoader,
    device: torch.device,
    config: Config,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    truth: List[float] = []
    prediction: List[float] = []
    row_keys: List[int] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        loss_dict = irony_model_loss(
            outputs, batch, config.irony_rationale_loss_weight
        )
        losses.append(float(loss_dict["loss"].item()))
        valid = torch.isfinite(batch["irony_scores"])
        truth.extend(batch["irony_scores"][valid].cpu().tolist())
        prediction.extend(torch.sigmoid(outputs["score_logits"])[valid].cpu().tolist())
        valid_indices = torch.where(valid.cpu())[0].tolist()
        row_keys.extend([raw_batch["row_keys"][i] for i in valid_indices])

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mae": float(mean_absolute_error(truth, prediction)) if truth else float("nan"),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5)
        if truth
        else float("nan"),
        "truth": np.asarray(truth, dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "row_keys": np.asarray(row_keys, dtype=np.int64),
    }


def train_irony_model(
    model: IronyRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Config,
) -> Tuple[IronyRegressor, Dict[str, Any]]:
    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        config.irony_learning_rate,
        config.irony_weight_decay,
        config.irony_epochs,
        len(train_loader),
        config.warmup_ratio,
    )
    scaler = make_grad_scaler(device, config.use_amp)
    best_state = cpu_state_dict(model)
    best_loss = float("inf")
    best_metrics: Dict[str, Any] = {}
    patience = 0

    for epoch in range(1, config.irony_epochs + 1):
        model.train()
        total_steps = len(train_loader)
        epoch_started_at = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        progress = tqdm(
            train_loader,
            desc=f"Irony BERT [{epoch}/{config.irony_epochs}]",
            leave=True,
            dynamic_ncols=True,
        )
        running = 0.0
        running_parts = {
            "regression": 0.0,
            "rationale": 0.0,
        }

        for step, raw_batch in enumerate(progress, start=1):
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, config.use_amp):
                outputs = model(**encoder_forward_kwargs(batch))
                loss_dict = irony_model_loss(
                    outputs, batch, config.irony_rationale_loss_weight
                )
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            for key in running_parts:
                running_parts[key] += float(loss_dict[key].item())

            average_parts = {
                key: value / step
                for key, value in running_parts.items()
            }
            _update_training_progress(
                progress,
                config=config,
                stage="irony",
                epoch=epoch,
                total_epochs=config.irony_epochs,
                step=step,
                total_steps=total_steps,
                running_loss=running,
                optimizer=optimizer,
                epoch_started_at=epoch_started_at,
                device=device,
                extra_losses=average_parts,
            )

        metrics = evaluate_irony_model(model, val_loader, device, config)
        print(
            f"[Irony] epoch={epoch} val_loss={metrics['loss']:.4f} "
            f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f}"
        )
        append_training_progress(
            config,
            stage="irony",
            status="validation",
            epoch=epoch,
            total_epochs=config.irony_epochs,
            step=len(train_loader),
            total_steps=len(train_loader),
            loss=metrics["loss"],
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.time() - epoch_started_at,
            eta_seconds=0.0,
            device=device,
            metrics=compact_scalar_metrics(metrics),
        )

        if metrics["loss"] < best_loss - 1e-6:
            best_loss = metrics["loss"]
            best_state = cpu_state_dict(model)
            best_metrics = compact_scalar_metrics(metrics)
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print("Irony BERT 조기 종료")
                break

    model.load_state_dict(best_state)
    model.to(device)
    return model, {"best": best_metrics}


# %% [markdown]
# ## 11. 문서 지수 계산과 근거 단어 추출

# %%
def _word_index_for_token(
    starts: Sequence[int],
    ends: Sequence[int],
    token_start: int,
    token_end: int,
) -> Optional[int]:
    if token_end <= token_start or not starts:
        return None
    candidate = bisect.bisect_right(starts, token_start) - 1
    for idx in (candidate, candidate + 1):
        if 0 <= idx < len(starts) and max(starts[idx], token_start) < min(ends[idx], token_end):
            return idx
    return None


def aggregate_token_probabilities_to_words(
    text: str,
    offsets_by_chunk: np.ndarray,
    probabilities_by_chunk: np.ndarray,
) -> List[Dict[str, Any]]:
    """겹치는 chunk/subword 확률을 공백 기준 어절별 최대 확률로 합칩니다."""
    words = whitespace_word_spans(text)
    if not words:
        return []
    starts = [w[0] for w in words]
    ends = [w[1] for w in words]
    scores = np.zeros(len(words), dtype=np.float32)

    for offsets, probabilities in zip(offsets_by_chunk, probabilities_by_chunk):
        for (start, end), probability in zip(offsets, probabilities):
            start, end = int(start), int(end)
            if start < 0 or end <= start:
                continue
            word_index = _word_index_for_token(starts, ends, start, end)
            if word_index is not None:
                scores[word_index] = max(scores[word_index], float(probability))

    return [
        {
            "word": word,
            "start": int(start),
            "end": int(end),
            "score": float(score),
        }
        for (start, end, word), score in zip(words, scores)
    ]


def word_ratio_from_predictions(word_predictions: Sequence[Mapping[str, Any]], threshold: float) -> float:
    if not word_predictions:
        return 0.0
    marked = sum(float(item["score"]) >= threshold for item in word_predictions)
    return float(marked / len(word_predictions))


def _top_word_evidence(
    word_predictions: Sequence[Mapping[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    ranked = sorted(word_predictions, key=lambda item: float(item["score"]), reverse=True)
    return [dict(item) for item in ranked[:top_k]]


@torch.no_grad()
def predict_index_scores(
    model: DualIndexBiLSTM,
    loader: DataLoader,
    device: torch.device,
    emotion_token_threshold: float,
    exaggeration_token_threshold: float,
    include_evidence: bool = False,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        emotion_prob = torch.sigmoid(outputs["emotion_logits"]).cpu().numpy()
        exaggeration_prob = torch.sigmoid(outputs["exaggeration_logits"]).cpu().numpy()
        offsets = raw_batch["offset_mapping"].cpu().numpy()
        chunk_to_doc = raw_batch["chunk_to_doc"].cpu().numpy()

        for doc_index, text in enumerate(raw_batch["texts"]):
            chunk_indices = np.where(chunk_to_doc == doc_index)[0]
            emotion_words = aggregate_token_probabilities_to_words(
                text, offsets[chunk_indices], emotion_prob[chunk_indices]
            )
            exag_words = aggregate_token_probabilities_to_words(
                text, offsets[chunk_indices], exaggeration_prob[chunk_indices]
            )
            record = {
                "row_key": int(raw_batch["row_keys"][doc_index]),
                "emotion_index": word_ratio_from_predictions(
                    emotion_words, emotion_token_threshold
                ),
                "exaggeration_index": word_ratio_from_predictions(
                    exag_words, exaggeration_token_threshold
                ),
            }
            if include_evidence:
                record["emotion_evidence"] = _top_word_evidence(emotion_words)
                record["exaggeration_evidence"] = _top_word_evidence(exag_words)
            records.append(record)

    return pd.DataFrame(records)


def attention_to_word_evidence(
    text: str,
    offsets_by_chunk: np.ndarray,
    token_attention_by_chunk: np.ndarray,
    chunk_attention: np.ndarray,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    weighted = token_attention_by_chunk * chunk_attention[:, None]
    words = aggregate_token_probabilities_to_words(text, offsets_by_chunk, weighted)
    return _top_word_evidence(words, top_k=top_k)


@torch.no_grad()
def predict_irony_scores(
    model: IronyRegressor,
    loader: DataLoader,
    device: torch.device,
    include_evidence: bool = False,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        scores = torch.sigmoid(outputs["score_logits"]).cpu().numpy()
        offsets = raw_batch["offset_mapping"].cpu().numpy()
        token_attention = outputs["token_attention"].cpu().numpy()
        chunk_attention = outputs["chunk_attention"].cpu().numpy()
        chunk_to_doc = raw_batch["chunk_to_doc"].cpu().numpy()

        for doc_index, text in enumerate(raw_batch["texts"]):
            record = {
                "row_key": int(raw_batch["row_keys"][doc_index]),
                "irony_index": float(scores[doc_index]),
            }
            if include_evidence:
                chunk_indices = np.where(chunk_to_doc == doc_index)[0]
                record["irony_evidence"] = attention_to_word_evidence(
                    text,
                    offsets[chunk_indices],
                    token_attention[chunk_indices],
                    chunk_attention[chunk_indices],
                )
            records.append(record)

    return pd.DataFrame(records)


@torch.no_grad()
def predict_stage1_scores(
    model: Stage1SuspicionModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    include_evidence: bool = False,
) -> pd.DataFrame:
    model.eval()
    records: List[Dict[str, Any]] = []

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(**encoder_forward_kwargs(batch))
        probabilities = torch.sigmoid(outputs["doc_logits"] / temperature).cpu().numpy()
        total_attention = torch.sigmoid(outputs["total_attention_logits"]).cpu().numpy()
        offsets = raw_batch["offset_mapping"].cpu().numpy()
        token_attention = outputs["token_attention"].cpu().numpy()
        chunk_attention = outputs["chunk_attention"].cpu().numpy()
        chunk_to_doc = raw_batch["chunk_to_doc"].cpu().numpy()

        for doc_index, text in enumerate(raw_batch["texts"]):
            record = {
                "row_key": int(raw_batch["row_keys"][doc_index]),
                "suspicious_probability": float(probabilities[doc_index]),
                "predicted_total_attention": float(total_attention[doc_index]),
            }
            if include_evidence:
                chunk_indices = np.where(chunk_to_doc == doc_index)[0]
                record["stage1_evidence"] = attention_to_word_evidence(
                    text,
                    offsets[chunk_indices],
                    token_attention[chunk_indices],
                    chunk_attention[chunk_indices],
                )
            records.append(record)

    return pd.DataFrame(records)


# %% [markdown]
# ## 12. 풍자/가짜 규칙 임계값 학습

# %%
def stage2_rule_prediction(
    emotion_index: np.ndarray,
    exaggeration_index: np.ndarray,
    irony_index: np.ndarray,
    rule: Mapping[str, float],
) -> np.ndarray:
    """
    사용자 정의 규칙:
    1) 과장과 아이러니가 함께 높으면 풍자
    2) 아이러니가 매우 높고 다른 지수보다 특히 높으면 풍자
    3) 그 외(감정자극만 높은 경우 포함)는 가짜
    """
    joint = (
        (exaggeration_index >= rule["exaggeration_threshold"])
        & (irony_index >= rule["irony_threshold"])
    )
    dominant_irony = (
        (irony_index >= rule["irony_high_threshold"])
        & (irony_index >= emotion_index + rule["irony_dominance_margin"])
        & (irony_index >= exaggeration_index + rule["irony_dominance_margin"])
    )
    return (joint | dominant_irony).astype(int)  # 1=satire, 0=fake


def _candidate_thresholds(values: np.ndarray, defaults: Sequence[float]) -> List[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    quantiles = np.quantile(values, np.linspace(0.10, 0.90, 9)).tolist() if len(values) else []
    candidates = sorted({_clip01(float(x)) for x in [*defaults, *quantiles]})
    return candidates


def tune_stage2_rule(validation_scores: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
    required = {"emotion_index", "exaggeration_index", "irony_index", "stage2_label"}
    missing = required - set(validation_scores.columns)
    if missing:
        raise ValueError(f"Stage 2 임계값 탐색 데이터에 열이 없습니다: {sorted(missing)}")

    e = validation_scores["emotion_index"].to_numpy(float)
    x = validation_scores["exaggeration_index"].to_numpy(float)
    i = validation_scores["irony_index"].to_numpy(float)
    y = validation_scores["stage2_label"].to_numpy(int)

    exag_candidates = _candidate_thresholds(x, [0.03, 0.05, 0.08, 0.10, 0.15, 0.20])
    irony_candidates = _candidate_thresholds(i, [0.35, 0.45, 0.55, 0.65, 0.75])
    high_candidates = _candidate_thresholds(i, [0.65, 0.75, 0.85, 0.90])
    margins = [0.0, 0.05, 0.10, 0.15, 0.20]

    best_rule: Optional[Dict[str, float]] = None
    best_metrics: Optional[Dict[str, Any]] = None

    for exag_t in exag_candidates:
        for irony_t in irony_candidates:
            for high_t in high_candidates:
                if high_t < irony_t:
                    continue
                for margin in margins:
                    rule = {
                        "exaggeration_threshold": float(exag_t),
                        "irony_threshold": float(irony_t),
                        "irony_high_threshold": float(high_t),
                        "irony_dominance_margin": float(margin),
                    }
                    pred = stage2_rule_prediction(e, x, i, rule)
                    metrics = {
                        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                        "satire_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
                        "accuracy": float(accuracy_score(y, pred)),
                        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                    }
                    key = (metrics["macro_f1"], metrics["satire_recall"], metrics["accuracy"])
                    current = (
                        best_metrics["macro_f1"],
                        best_metrics["satire_recall"],
                        best_metrics["accuracy"],
                    ) if best_metrics else (-1.0, -1.0, -1.0)
                    if key > current:
                        best_rule = rule
                        best_metrics = metrics

    if best_rule is None or best_metrics is None:
        raise RuntimeError("Stage 2 규칙 임계값을 선택하지 못했습니다.")

    # 감정만 높은 경우를 설명할 때 쓰는 기준이며 최종 풍자 규칙에는 직접 사용하지 않습니다.
    best_rule["emotion_high_threshold"] = float(
        max(0.05, np.quantile(e, 0.75) if len(e) else 0.15)
    )
    return best_rule, best_metrics


def stage2_reason(
    emotion: float,
    exaggeration: float,
    irony: float,
    rule: Mapping[str, float],
) -> str:
    joint = exaggeration >= rule["exaggeration_threshold"] and irony >= rule["irony_threshold"]
    dominant = (
        irony >= rule["irony_high_threshold"]
        and irony >= emotion + rule["irony_dominance_margin"]
        and irony >= exaggeration + rule["irony_dominance_margin"]
    )
    emotion_only = (
        emotion >= rule.get("emotion_high_threshold", 0.15)
        and exaggeration < rule["exaggeration_threshold"]
        and irony < rule["irony_threshold"]
    )

    if joint:
        return "과장 지수와 아이러니 지수가 함께 높음"
    if dominant:
        return "아이러니 지수가 매우 높고 다른 지수보다 두드러짐"
    if emotion_only:
        return "감정자극 지수만 높아 풍자보다 가짜뉴스 규칙에 가까움"
    return "풍자 조건을 충족하지 않아 가짜뉴스로 분류"


# %% [markdown]
# ## 13. 저장과 로딩

# %%
def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(dict(data)), file, ensure_ascii=False, indent=2)


def safe_torch_load(path: Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_pipeline_bundle(
    output_dir: str,
    tokenizer,
    stage1_model: Stage1SuspicionModel,
    index_model: DualIndexBiLSTM,
    irony_model: IronyRegressor,
    config: Config,
    stage1_temperature: float,
    stage1_threshold_info: Mapping[str, Any],
    emotion_token_threshold: Mapping[str, Any],
    exaggeration_token_threshold: Mapping[str, Any],
    stage2_rule: Mapping[str, Any],
    training_info: Optional[Mapping[str, Any]] = None,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(root / "tokenizer")
    stage1_model.document_encoder.encoder.config.save_pretrained(root / "encoder_config")
    torch.save(cpu_state_dict(stage1_model), root / "stage1_model.pt")
    torch.save(cpu_state_dict(index_model), root / "index_bilstm.pt")
    torch.save(cpu_state_dict(irony_model), root / "irony_model.pt")

    metadata = {
        "config": asdict(config),
        "stage1_temperature": float(stage1_temperature),
        "stage1_threshold": dict(stage1_threshold_info),
        "emotion_token_threshold": dict(emotion_token_threshold),
        "exaggeration_token_threshold": dict(exaggeration_token_threshold),
        "stage2_rule": dict(stage2_rule),
        "index_model_config": index_model.model_config,
        "training_info": dict(training_info or {}),
        "label_mapping": KOREAN_LABELS,
    }
    save_json(root / "metadata.json", metadata)
    return root


# %% [markdown]
# ## 14. 실제 판독용 파이프라인

# %%
class FakeNewsDetector:
    """저장된 세 모델과 임계값을 로드해 새 기사를 판독합니다."""

    def __init__(
        self,
        root: Path,
        tokenizer,
        stage1_model: Stage1SuspicionModel,
        index_model: DualIndexBiLSTM,
        irony_model: IronyRegressor,
        metadata: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.root = root
        self.tokenizer = tokenizer
        self.stage1_model = stage1_model.eval()
        self.index_model = index_model.eval()
        self.irony_model = irony_model.eval()
        self.metadata = metadata
        self.device = device

    @classmethod
    def load(
        cls,
        output_dir: str,
        device: Optional[torch.device] = None,
    ) -> "FakeNewsDetector":
        root = Path(output_dir)
        if not root.exists():
            raise FileNotFoundError(root)
        device = device or get_device()

        with (root / "metadata.json").open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", use_fast=True)
        encoder_config = AutoConfig.from_pretrained(root / "encoder_config")
        cfg = metadata["config"]

        stage1_model = Stage1SuspicionModel(
            encoder_config=encoder_config,
            pretrained=False,
            dropout=float(cfg["bert_head_dropout"]),
        )
        irony_model = IronyRegressor(
            encoder_config=copy.deepcopy(encoder_config),
            pretrained=False,
            dropout=float(cfg["bert_head_dropout"]),
        )
        index_model = DualIndexBiLSTM(**metadata["index_model_config"])

        stage1_model.load_state_dict(
            safe_torch_load(root / "stage1_model.pt", map_location="cpu")
        )
        index_model.load_state_dict(
            safe_torch_load(root / "index_bilstm.pt", map_location="cpu")
        )
        irony_model.load_state_dict(
            safe_torch_load(root / "irony_model.pt", map_location="cpu")
        )

        stage1_model.to(device)
        index_model.to(device)
        irony_model.to(device)
        return cls(
            root,
            tokenizer,
            stage1_model,
            index_model,
            irony_model,
            metadata,
            device,
        )

    def _prediction_frame(self, text: str) -> pd.DataFrame:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("판독할 뉴스 text는 비어 있지 않은 문자열이어야 합니다.")
        return pd.DataFrame(
            [
                {
                    "id": "inference-0",
                    "row_key": 0,
                    "text": text.strip(),
                    "label": "진짜뉴스",  # Dataset 인터페이스용 placeholder
                    "label_norm": "real",
                    "total_attention": np.nan,
                    "token_attention": np.nan,
                    "emotion_spans": np.nan,
                    "exaggeration_spans": np.nan,
                    "irony_spans": np.nan,
                    "emotion_score": np.nan,
                    "exaggeration_score": np.nan,
                    "irony_score": np.nan,
                }
            ]
        )

    @torch.no_grad()
    def predict(self, text: str, include_evidence: bool = True) -> Dict[str, Any]:
        cfg = self.metadata["config"]
        frame = self._prediction_frame(text)

        # Stage 1
        stage1_ds = Stage1DocumentDataset(
            frame,
            self.tokenizer,
            int(cfg["max_length"]),
            int(cfg["bert_stride"]),
        )
        stage1_loader = DataLoader(
            stage1_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=Stage1Collator(self.tokenizer),
        )
        stage1_result = predict_stage1_scores(
            self.stage1_model,
            stage1_loader,
            self.device,
            float(self.metadata["stage1_temperature"]),
            include_evidence=include_evidence,
        ).iloc[0].to_dict()

        suspicious_probability = float(stage1_result["suspicious_probability"])
        suspicious_threshold = float(self.metadata["stage1_threshold"]["threshold"])
        result: Dict[str, Any] = {
            "final_label": "진짜뉴스",
            "suspicious_probability": suspicious_probability,
            "suspicious_threshold": suspicious_threshold,
            "predicted_total_attention": float(
                stage1_result["predicted_total_attention"]
            ),
            "stage1_passed": suspicious_probability >= suspicious_threshold,
        }
        if include_evidence:
            result["stage1_evidence"] = stage1_result.get("stage1_evidence", [])

        # 의심 확률이 임계값보다 낮으면 진짜뉴스로 종료합니다.
        if suspicious_probability < suspicious_threshold:
            result["decision_reason"] = "의심 뉴스 확률이 Stage 1 임계값보다 낮음"
            return result

        # Stage 2-A: 감정/과장 지수
        index_ds = IndexDocumentDataset(
            frame,
            self.tokenizer,
            int(cfg["max_length"]),
            int(cfg["index_stride"]),
            require_targets=False,
        )
        index_loader = DataLoader(
            index_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=IndexCollator(self.tokenizer),
        )
        index_result = predict_index_scores(
            self.index_model,
            index_loader,
            self.device,
            float(self.metadata["emotion_token_threshold"]["threshold"]),
            float(self.metadata["exaggeration_token_threshold"]["threshold"]),
            include_evidence=include_evidence,
        ).iloc[0].to_dict()

        # Stage 2-B: 아이러니 지수
        irony_ds = IronyDocumentDataset(
            frame,
            self.tokenizer,
            int(cfg["max_length"]),
            int(cfg["bert_stride"]),
            require_targets=False,
        )
        irony_loader = DataLoader(
            irony_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=IronyCollator(self.tokenizer),
        )
        irony_result = predict_irony_scores(
            self.irony_model,
            irony_loader,
            self.device,
            include_evidence=include_evidence,
        ).iloc[0].to_dict()

        emotion = float(index_result["emotion_index"])
        exaggeration = float(index_result["exaggeration_index"])
        irony = float(irony_result["irony_index"])
        rule = self.metadata["stage2_rule"]
        satire = bool(
            stage2_rule_prediction(
                np.asarray([emotion]),
                np.asarray([exaggeration]),
                np.asarray([irony]),
                rule,
            )[0]
        )

        result.update(
            {
                "final_label": "풍자뉴스" if satire else "가짜뉴스",
                "emotion_index": emotion,
                "exaggeration_index": exaggeration,
                "irony_index": irony,
                "decision_reason": stage2_reason(
                    emotion, exaggeration, irony, rule
                ),
                "stage2_rule": rule,
            }
        )
        if include_evidence:
            result["emotion_evidence"] = index_result.get("emotion_evidence", [])
            result["exaggeration_evidence"] = index_result.get(
                "exaggeration_evidence", []
            )
            result["irony_evidence"] = irony_result.get("irony_evidence", [])
        return result

    def predict_many(
        self,
        texts: Sequence[str],
        include_evidence: bool = False,
    ) -> pd.DataFrame:
        """간단하고 안전한 순차 배치 API. 대규모 서비스에서는 DataLoader 배치화를 권장합니다."""
        records = [self.predict(text, include_evidence=include_evidence) for text in texts]
        return pd.DataFrame(records)


# %% [markdown]
# ## 15. 전체 학습 실행 함수

# %%
def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    config: Config,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def _prepare_stage2_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label_norm"].isin(["fake", "satire"])].reset_index(drop=True)


def build_stage2_validation_scores(
    validation_df: pd.DataFrame,
    index_model: DualIndexBiLSTM,
    irony_model: IronyRegressor,
    tokenizer,
    device: torch.device,
    config: Config,
    emotion_token_threshold: float,
    exaggeration_token_threshold: float,
) -> pd.DataFrame:
    stage2_df = _prepare_stage2_frame(validation_df)
    index_ds = IndexDocumentDataset(
        stage2_df,
        tokenizer,
        config.max_length,
        config.index_stride,
        require_targets=False,
    )
    irony_ds = IronyDocumentDataset(
        stage2_df,
        tokenizer,
        config.max_length,
        config.bert_stride,
        require_targets=False,
    )
    index_loader = _make_loader(
        index_ds,
        config.index_batch_size,
        False,
        IndexCollator(tokenizer),
        config,
    )
    irony_loader = _make_loader(
        irony_ds,
        config.irony_batch_size,
        False,
        IronyCollator(tokenizer),
        config,
    )

    # 메모리가 작은 GPU에서도 동작하도록 두 모델을 순차적으로 올립니다.
    index_model.to(device)
    index_scores = predict_index_scores(
        index_model,
        index_loader,
        device,
        emotion_token_threshold,
        exaggeration_token_threshold,
    )
    index_model.to("cpu")
    clear_accelerator_cache()

    irony_model.to(device)
    irony_scores = predict_irony_scores(irony_model, irony_loader, device)
    irony_model.to("cpu")
    clear_accelerator_cache()
    labels = stage2_df[["row_key", "label_norm"]].copy()
    labels["stage2_label"] = (labels["label_norm"] == "satire").astype(int)
    return labels.merge(index_scores, on="row_key").merge(irony_scores, on="row_key")


def evaluate_end_to_end(
    test_df: pd.DataFrame,
    tokenizer,
    stage1_model: Stage1SuspicionModel,
    index_model: DualIndexBiLSTM,
    irony_model: IronyRegressor,
    device: torch.device,
    config: Config,
    temperature: float,
    suspicion_threshold: float,
    emotion_token_threshold: float,
    exaggeration_token_threshold: float,
    stage2_rule: Mapping[str, float],
) -> Dict[str, Any]:
    stage1_ds = Stage1DocumentDataset(
        test_df, tokenizer, config.max_length, config.bert_stride
    )
    index_ds = IndexDocumentDataset(
        test_df,
        tokenizer,
        config.max_length,
        config.index_stride,
        require_targets=False,
    )
    irony_ds = IronyDocumentDataset(
        test_df,
        tokenizer,
        config.max_length,
        config.bert_stride,
        require_targets=False,
    )

    stage1_model.to(device)
    stage1_scores = predict_stage1_scores(
        stage1_model,
        _make_loader(
            stage1_ds,
            config.stage1_batch_size,
            False,
            Stage1Collator(tokenizer),
            config,
        ),
        device,
        temperature,
    )
    stage1_model.to("cpu")
    clear_accelerator_cache()

    index_model.to(device)
    index_scores = predict_index_scores(
        index_model,
        _make_loader(
            index_ds,
            config.index_batch_size,
            False,
            IndexCollator(tokenizer),
            config,
        ),
        device,
        emotion_token_threshold,
        exaggeration_token_threshold,
    )
    index_model.to("cpu")
    clear_accelerator_cache()

    irony_model.to(device)
    irony_scores = predict_irony_scores(
        irony_model,
        _make_loader(
            irony_ds,
            config.irony_batch_size,
            False,
            IronyCollator(tokenizer),
            config,
        ),
        device,
    )
    irony_model.to("cpu")
    clear_accelerator_cache()

    scored = (
        test_df[["row_key", "label_norm"]]
        .merge(stage1_scores, on="row_key")
        .merge(index_scores, on="row_key")
        .merge(irony_scores, on="row_key")
    )
    suspicious = scored["suspicious_probability"].to_numpy() >= suspicion_threshold
    stage2_pred = stage2_rule_prediction(
        scored["emotion_index"].to_numpy(),
        scored["exaggeration_index"].to_numpy(),
        scored["irony_index"].to_numpy(),
        stage2_rule,
    )
    predicted = np.where(
        ~suspicious,
        "real",
        np.where(stage2_pred == 1, "satire", "fake"),
    )
    truth = scored["label_norm"].to_numpy()

    report = classification_report(
        truth,
        predicted,
        labels=["real", "fake", "satire"],
        target_names=["진짜뉴스", "가짜뉴스", "풍자뉴스"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        truth, predicted, labels=["real", "fake", "satire"]
    ).tolist()
    scored["prediction"] = predicted
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": ["real", "fake", "satire"],
        "confusion_matrix": matrix,
        "predictions": scored,
    }


def run_full_training(config: Optional[Config] = None) -> Dict[str, Any]:
    """데이터 로드부터 세 모델 학습, 임계값 튜닝, 저장, 테스트까지 수행합니다."""
    config = config or Config()
    set_seed(config.random_seed)
    device = get_device()
    print(f"파이프라인 버전: {PIPELINE_VERSION}")
    print(f"사용 장치: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    progress_path = initialize_training_progress(config)
    print(f"진행 기록: {progress_path.resolve()}")
    run_started_at = time.time()
    append_training_progress(
        config,
        stage="pipeline",
        status="started",
        device=device,
    )

    df = load_news_dataframe(config)
    train_df, val_df, test_df = split_dataframe(df, config)
    print_split_summary(train_df, val_df, test_df)

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            "문자 구간 attention 정렬을 위해 fast tokenizer가 필요합니다."
        )

    # -------------------- Stage 1 --------------------
    print("\n[1/5] Stage 1: 진짜뉴스 대 의심뉴스 학습 시작")
    stage1_train_ds = Stage1DocumentDataset(
        train_df, tokenizer, config.max_length, config.bert_stride
    )
    stage1_val_ds = Stage1DocumentDataset(
        val_df, tokenizer, config.max_length, config.bert_stride
    )
    stage1_train_loader = _make_loader(
        stage1_train_ds,
        config.stage1_batch_size,
        True,
        Stage1Collator(tokenizer),
        config,
    )
    stage1_val_loader = _make_loader(
        stage1_val_ds,
        config.stage1_batch_size,
        False,
        Stage1Collator(tokenizer),
        config,
    )
    stage1_model = Stage1SuspicionModel(
        model_name=config.base_model_name,
        pretrained=True,
        dropout=config.bert_head_dropout,
    ).to(device)
    stage1_model, stage1_info = train_stage1(
        stage1_model,
        stage1_train_loader,
        stage1_val_loader,
        train_df,
        device,
        config,
    )

    # 최적 모델로 검증 logit을 다시 계산한 뒤 확률을 보정합니다.
    pos_weight = torch.tensor(
        stage1_info["positive_weight"], dtype=torch.float32, device=device
    )
    stage1_val_metrics = evaluate_stage1(
        stage1_model,
        stage1_val_loader,
        device,
        pos_weight,
        config,
    )
    temperature_scaler = TemperatureScaler().fit(
        stage1_val_metrics["logits"], stage1_val_metrics["labels"]
    )
    calibrated_probabilities = temperature_scaler.transform(stage1_val_metrics["logits"])
    stage1_threshold = choose_suspicion_threshold(
        stage1_val_metrics["labels"],
        calibrated_probabilities,
        config.target_suspicious_recall,
    )
    print("Stage 1 temperature:", temperature_scaler.temperature)
    print("Stage 1 threshold:", stage1_threshold)
    stage1_model.to("cpu")
    clear_accelerator_cache()

    # -------------------- Stage 2 학습 데이터 --------------------
    print("\n[2/5] Stage 2 학습 데이터 준비")
    stage2_train_df = _prepare_stage2_frame(train_df)
    stage2_val_df = _prepare_stage2_frame(val_df)
    if len(stage2_train_df) == 0 or len(stage2_val_df) == 0:
        raise ValueError("가짜뉴스/풍자뉴스가 train과 validation에 모두 필요합니다.")
    if stage2_train_df["label_norm"].nunique() < 2:
        raise ValueError("Stage 2 train에는 가짜뉴스와 풍자뉴스가 모두 필요합니다.")
    if stage2_val_df["label_norm"].nunique() < 2:
        warnings.warn("Stage 2 validation에 한 라벨만 있어 규칙 튜닝이 불안정할 수 있습니다.")

    # -------------------- Bi-LSTM --------------------
    print("\n[3/5] Bi-LSTM: 감정자극·과장 지수 학습 시작")

    index_train_df, index_val_df = prepare_index_training_frames(
        stage2_train_df,
        stage2_val_df,
        config,
    )

    index_train_ds = IndexDocumentDataset(
        index_train_df,
        tokenizer,
        config.max_length,
        config.index_stride,
        require_targets=True,
    )
    index_val_ds = IndexDocumentDataset(
        index_val_df,
        tokenizer,
        config.max_length,
        config.index_stride,
        require_targets=True,
    )
    index_train_loader = _make_loader(
        index_train_ds,
        config.index_batch_size,
        True,
        IndexCollator(tokenizer),
        config,
    )
    index_val_loader = _make_loader(
        index_val_ds,
        config.index_batch_size,
        False,
        IndexCollator(tokenizer),
        config,
    )
    padding_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    index_model = DualIndexBiLSTM(
        vocab_size=len(tokenizer),
        padding_idx=padding_idx,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.lstm_hidden_dim,
        num_layers=config.lstm_layers,
        dropout=config.index_dropout,
    ).to(device)
    index_model, index_info = train_index_model(
        index_model,
        index_train_loader,
        index_val_loader,
        index_train_ds,
        device,
        config,
    )

    e_weight = torch.tensor(
        index_info["emotion_positive_weight"], dtype=torch.float32, device=device
    )
    x_weight = torch.tensor(
        index_info["exaggeration_positive_weight"], dtype=torch.float32, device=device
    )
    index_val_metrics = evaluate_index_model(
        index_model,
        index_val_loader,
        device,
        e_weight,
        x_weight,
        config,
    )
    emotion_token_threshold = choose_token_threshold(
        index_val_metrics["emotion_token_true"],
        index_val_metrics["emotion_token_prob"],
    )
    exaggeration_token_threshold = choose_token_threshold(
        index_val_metrics["exaggeration_token_true"],
        index_val_metrics["exaggeration_token_prob"],
    )
    print("Emotion token threshold:", emotion_token_threshold)
    print("Exaggeration token threshold:", exaggeration_token_threshold)
    index_model.to("cpu")
    clear_accelerator_cache()

    # -------------------- Irony BERT --------------------
    print("\n[4/5] Irony BERT: 아이러니 지수 학습 시작")

    # irony_score가 없는 행은 아이러니 회귀 모델 학습에서만 제외합니다.
    irony_train_df = stage2_train_df[
        stage2_train_df["irony_score"].notna()
    ].copy()
    irony_val_df = stage2_val_df[
        stage2_val_df["irony_score"].notna()
    ].copy()

    print(
        "Irony 학습 데이터:",
        f"train={len(irony_train_df)}/{len(stage2_train_df)},",
        f"validation={len(irony_val_df)}/{len(stage2_val_df)}",
    )

    if len(irony_train_df) == 0:
        raise ValueError(
            "유효한 irony_score가 있는 아이러니 학습 데이터가 없습니다."
        )
    if len(irony_val_df) == 0:
        raise ValueError(
            "유효한 irony_score가 있는 아이러니 validation 데이터가 없습니다."
        )

    irony_train_ds = IronyDocumentDataset(
        irony_train_df,
        tokenizer,
        config.max_length,
        config.bert_stride,
        require_targets=True,
    )
    irony_val_ds = IronyDocumentDataset(
        irony_val_df,
        tokenizer,
        config.max_length,
        config.bert_stride,
        require_targets=True,
    )
    irony_train_loader = _make_loader(
        irony_train_ds,
        config.irony_batch_size,
        True,
        IronyCollator(tokenizer),
        config,
    )
    irony_val_loader = _make_loader(
        irony_val_ds,
        config.irony_batch_size,
        False,
        IronyCollator(tokenizer),
        config,
    )
    irony_model = IronyRegressor(
        model_name=config.base_model_name,
        pretrained=True,
        dropout=config.bert_head_dropout,
    ).to(device)
    irony_model, irony_info = train_irony_model(
        irony_model,
        irony_train_loader,
        irony_val_loader,
        device,
        config,
    )
    irony_model.to("cpu")
    clear_accelerator_cache()

    # -------------------- Stage 2 규칙 임계값 튜닝 --------------------
    print("\n[5/5] 임계값 튜닝·모델 저장·최종 테스트")
    validation_scores = build_stage2_validation_scores(
        val_df,
        index_model,
        irony_model,
        tokenizer,
        device,
        config,
        emotion_token_threshold["threshold"],
        exaggeration_token_threshold["threshold"],
    )
    stage2_rule, stage2_rule_metrics = tune_stage2_rule(validation_scores)
    print("Stage 2 rule:", stage2_rule)
    print("Stage 2 validation metrics:", stage2_rule_metrics)

    # -------------------- 저장 --------------------
    training_info = {
        "stage1": stage1_info,
        "index_model": index_info,
        "irony": irony_info,
        "stage2_rule_validation": stage2_rule_metrics,
    }
    bundle_path = save_pipeline_bundle(
        config.output_dir,
        tokenizer,
        stage1_model,
        index_model,
        irony_model,
        config,
        temperature_scaler.temperature,
        stage1_threshold,
        emotion_token_threshold,
        exaggeration_token_threshold,
        stage2_rule,
        training_info,
    )
    print(f"저장 완료: {bundle_path.resolve()}")

    # -------------------- 최종 테스트 --------------------
    test_metrics = evaluate_end_to_end(
        test_df,
        tokenizer,
        stage1_model,
        index_model,
        irony_model,
        device,
        config,
        temperature_scaler.temperature,
        stage1_threshold["threshold"],
        emotion_token_threshold["threshold"],
        exaggeration_token_threshold["threshold"],
        stage2_rule,
    )
    print(
        f"End-to-end test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro-F1={test_metrics['macro_f1']:.4f}"
    )
    print("Confusion matrix [real, fake, satire]:")
    print(np.asarray(test_metrics["confusion_matrix"]))
    total_elapsed = time.time() - run_started_at
    print(f"전체 학습 소요 시간: {_format_duration(total_elapsed)}")
    append_training_progress(
        config,
        stage="pipeline",
        status="completed",
        loss=float("nan"),
        elapsed_seconds=total_elapsed,
        eta_seconds=0.0,
        device=device,
        metrics={
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "bundle_path": str(bundle_path),
        },
    )

    return {
        "config": config,
        "dataframe": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "tokenizer": tokenizer,
        "stage1_model": stage1_model,
        "index_model": index_model,
        "irony_model": irony_model,
        "temperature": temperature_scaler.temperature,
        "stage1_threshold": stage1_threshold,
        "emotion_token_threshold": emotion_token_threshold,
        "exaggeration_token_threshold": exaggeration_token_threshold,
        "stage2_rule": stage2_rule,
        "test_metrics": test_metrics,
        "bundle_path": bundle_path,
    }


# %% [markdown]
# ## 16. 사용 예시
#
# Jupyter에서 다음 코드를 별도 셀로 실행합니다.
#
# ```python
# config = Config(
#     data_path="news_dataset.csv",
#     output_dir="artifacts/korean_fake_news_pipeline",
#     stage1_epochs=4,
#     index_epochs=6,
#     irony_epochs=4,
# )
# artifacts = run_full_training(config)
#
# detector = FakeNewsDetector.load(config.output_dir)
# result = detector.predict("판독할 뉴스 제목과 본문")
# result
# ```
