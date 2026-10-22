# -*- coding: utf-8 -*-
"""
CMM-SCREEN：社区居民三高共病风险预测网页工具

运行前安装：
    python -m pip install streamlit pandas numpy xgboost scikit-learn imbalanced-learn openpyxl

请把 cmm_xgboost_bundle.pkl 放在 model/ 文件夹，然后运行：
    python -m streamlit run app.py

模型结局方向：
    Outcome = 1：三高共病高风险
    Outcome = 0：三高共病低风险
"""

from __future__ import annotations

import io
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# 1. 网页与模型文件设置
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
BUNDLE_FILENAME = "cmm_xgboost_bundle.pkl"
MODEL_DIR = APP_DIR / "model"
BUNDLE_PATH = MODEL_DIR / BUNDLE_FILENAME

# 批量预测结果展示列
BATCH_RESULT_DISPLAY_COLUMNS = [
    "风险等级(Risk level)",
    "高风险概率(Probability of high risk)",
    "低风险概率(Probability of low risk)",
]

BATCH_IDENTIFIER_ALIASES = [
    "对象编号(ID)", "对象编号", "唯一ID", "患者ID", "患者编号", "ID", "编号",
]


# =============================================================================
# 2. 变量名称、选项及数字编码
# =============================================================================
def option(label: str, code: float, *aliases: str) -> dict[str, Any]:
    return {"label": label, "code": code, "aliases": list(aliases)}


def yes_no_options() -> list[dict[str, Any]]:
    return [
        option("否（No）", 0, "否", "无", "没有", "No"),
        option("是（Yes）", 1, "是", "有", "Yes"),
    ]


# 特征分组
FEATURE_GROUPS = {
    "基本信息": ["年龄", "性别", "BMI", "学历", "收入", "职业", "婚姻", "睡眠异常"],
    "生活方式": ["吸烟", "饮酒", "饮食是否清淡", "主食习惯", "蔬菜摄入频率",
                 "肉摄入频率", "水果摄入频率", "乳制品摄入频率", "高脂高糖加工食品"],
    "中医体质": ["平和质得分", "气虚质得分", "阳虚质得分", "阴虚质得分",
                 "痰湿质得分", "湿热质得分", "血瘀质得分", "气郁质得分", "特禀质得分"],
    "实验室检查": ["WBC", "PLT", "MPV", "RDW", "NLR", "MLR", "PLR", "PNR",
                   "pH", "尿比重", "尿蛋白", "尿潜血"],
    "左脉参数": ["左h1", "左h2", "左h3", "左h4", "左h5", "左h3/h1", "左h4/h1", "左h5/h1",
                 "左t", "左t1", "左t2", "左t3", "左t4", "左t5", "左t1/t", "左t4/t5",
                 "左w1", "左w2", "左w1/t", "左a_s", "左a_d",
                 "左脉力", "左脉位", "左脉势", "左脉律", "左脉率"],
    "右脉参数": ["右h1", "右h2", "右h3", "右h4", "右h5", "右h3/h1", "右h4/h1", "右h5/h1",
                 "右t", "右t1", "右t2", "右t3", "右t4", "右t5", "右t1/t", "右t4/t5",
                 "右w1", "右w2", "右w1/t", "右a_s", "右a_d",
                 "右脉力", "右脉位", "右脉势", "右脉律", "右脉率"],
}

# 脉象分类变量选项
PULSE_FORCE_OPTIONS = [
    option("正常（Normal）", "正常", "正常", "Normal"),
    option("有力（Forceful）", "有力", "有力", "Forceful"),
    option("无力（Weak）", "无力", "无力", "Weak"),
]
PULSE_POSITION_OPTIONS = [
    option("中（Middle）", "中", "中", "Middle"),
    option("浮（Superficial）", "浮", "浮", "Superficial"),
    option("沉（Deep）", "沉", "沉", "Deep"),
]
PULSE_QUALITY_OPTIONS = [
    option("正常（Normal）", "正常", "正常", "Normal"),
    option("弦（Wiry）", "弦", "弦", "Wiry"),
    option("滑（Slippery）", "滑", "滑", "Slippery"),
    option("涩（Choppy）", "涩", "涩", "Choppy"),
    option("紧（Tight）", "紧", "紧", "Tight"),
]
PULSE_RHYTHM_OPTIONS = [
    option("齐（Regular）", "齐", "齐", "Regular"),
    option("不齐（Irregular）", "不齐", "不齐", "Irregular"),
]
PULSE_RATE_OPTIONS = [
    option("中（Moderate）", "中", "中", "Moderate"),
    option("缓（Slow）", "缓", "缓", "Slow"),
    option("迟（Very slow）", "迟", "迟", "Very slow"),
    option("慢（Slow）", "慢", "慢", "Slow"),
    option("数（Rapid）", "数", "数", "Rapid"),
    option("快（Fast）", "快", "快", "Fast"),
    option("疾（Very rapid）", "疾", "疾", "Very rapid"),
]
PULSE_NAME_OPTIONS = [
    option("平脉（Normal）", "平脉", "平脉", "Normal"),
    option("浮脉（Superficial）", "浮脉", "浮脉", "Superficial"),
    option("沉脉（Deep）", "沉脉", "沉脉", "Deep"),
    option("迟脉（Slow）", "迟脉", "迟脉", "Slow"),
    option("数脉（Rapid）", "数脉", "数脉", "Rapid"),
    option("滑脉（Slippery）", "滑脉", "滑脉", "Slippery"),
    option("涩脉（Choppy）", "涩脉", "涩脉", "Choppy"),
    option("虚脉（Weak）", "虚脉", "虚脉", "Weak"),
    option("实脉（Forceful）", "实脉", "实脉", "Forceful"),
    option("弦脉（Wiry）", "弦脉", "弦脉", "Wiry"),
    option("紧脉（Tight）", "紧脉", "紧脉", "Tight"),
    option("缓脉（Moderate）", "缓脉", "缓脉", "Moderate"),
    option("结脉（Irregular）", "结脉", "结脉", "Irregular"),
]

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    # 基本信息
    "年龄": {"label": "年龄（Age, years）", "batch_column": "年龄(Age)", "kind": "numeric", "aliases": ["年龄", "Age"]},
    "性别": {"label": "性别（Gender）", "batch_column": "性别(Gender)", "kind": "category", "aliases": ["性别", "Gender"],
             "options": [option("男（Male）", 1, "男", "男性", "Male"), option("女（Female）", 2, "女", "女性", "Female")]},
    "BMI": {"label": "BMI（kg/m²）", "batch_column": "BMI", "kind": "numeric", "aliases": ["BMI", "体质指数"]},
    "学历": {"label": "学历（Education）", "batch_column": "学历(Education)", "kind": "category", "aliases": ["学历", "文化程度", "Education"],
             "options": [option("小学及以下", 1, "小学及以下", "文盲"), option("初中", 2, "初中"),
                        option("高中/中专", 3, "高中", "中专"), option("大专", 4, "大专"),
                        option("本科", 5, "本科"), option("硕士及以上", 6, "硕士", "博士", "研究生")]},
    "收入": {"label": "月收入（Monthly income）", "batch_column": "月收入(Income)", "kind": "category", "aliases": ["收入", "月收入", "Income"],
             "options": [option("<3000元", 1, "<3000", "3000以下"), option("3000-5000元", 2, "3000-5000"),
                        option("5000-8000元", 3, "5000-8000"), option("8000-12000元", 4, "8000-12000"),
                        option(">12000元", 5, ">12000", "12000以上")]},
    "职业": {"label": "职业（Occupation）", "batch_column": "职业(Occupation)", "kind": "category", "aliases": ["职业", "Occupation"],
             "options": [option("机关事业单位", 1, "机关", "事业单位", "政府"), option("企业", 2, "企业"),
                        option("个体经营", 3, "个体"), option("农民", 4, "农民", "农业"),
                        option("工人", 5, "工人"), option("学生", 6, "学生"),
                        option("退休", 7, "退休"), option("无业", 8, "无业", "失业"),
                        option("其他", 9, "其他")]},
    "婚姻": {"label": "婚姻状况（Marital status）", "batch_column": "婚姻(Marital status)", "kind": "category", "aliases": ["婚姻", "婚姻状况", "Marital status"],
             "options": [option("未婚", 1, "未婚"), option("已婚", 2, "已婚", "初婚", "再婚"),
                        option("离异", 3, "离异", "离婚"), option("丧偶", 4, "丧偶")]},
    "睡眠异常": {"label": "睡眠异常（Sleep abnormality）", "batch_column": "睡眠异常(Sleep abnormality)", "kind": "category", "aliases": ["睡眠异常", "睡眠", "Sleep"],
                 "options": [option("正常", 1, "正常", "无"), option("异常", 2, "异常", "有")]},
    # 生活方式
    "吸烟": {"label": "吸烟情况（Smoking）", "batch_column": "吸烟(Smoking)", "kind": "category", "aliases": ["吸烟", "Smoking"],
             "options": [option("从不", 1, "从不", "不吸", "Never"), option("已戒烟", 2, "已戒烟", "过去吸", "Former"),
                        option("吸烟", 3, "吸烟", "现在吸", "Current")]},
    "饮酒": {"label": "饮酒情况（Drinking）", "batch_column": "饮酒(Drinking)", "kind": "category", "aliases": ["饮酒", "喝酒", "Drinking"],
             "options": [option("从不", 1, "从不", "不饮", "Never"), option("饮酒", 2, "饮酒", "现在饮", "Current")]},
    "饮食是否清淡": {"label": "饮食是否清淡（Diet lightness）", "batch_column": "饮食清淡(Diet lightness)", "kind": "category", "aliases": ["饮食是否清淡", "饮食清淡", "Diet"],
                    "options": [option("清淡", 1, "清淡"), option("一般", 2, "一般"), option("油腻", 3, "油腻")]},
    "主食习惯": {"label": "主食习惯（Staple food）", "batch_column": "主食习惯(Staple food)", "kind": "category", "aliases": ["主食习惯", "主食", "Staple"],
                 "options": [option("精米白面为主", 1, "精米白面", "细粮"), option("粗细搭配", 2, "粗细搭配"),
                            option("粗粮为主", 3, "粗粮"), option("其他", 4, "其他")]},
    "蔬菜摄入频率": {"label": "蔬菜摄入频率（Vegetable intake）", "batch_column": "蔬菜摄入频率(Vegetable intake)", "kind": "category", "aliases": ["蔬菜摄入频率", "蔬菜", "Vegetable"],
                    "options": [option("很少/从不", 1, "很少", "从不", "Rarely"), option("偶尔", 2, "偶尔", "Sometimes"),
                               option("经常", 3, "经常", "Often"), option("每天", 4, "每天", "Daily")]},
    "肉摄入频率": {"label": "肉类摄入频率（Meat intake）", "batch_column": "肉摄入频率(Meat intake)", "kind": "category", "aliases": ["肉摄入频率", "肉类", "Meat"],
                  "options": [option("很少/从不", 1, "很少", "从不"), option("偶尔", 2, "偶尔"),
                             option("经常", 3, "经常"), option("每天", 4, "每天")]},
    "水果摄入频率": {"label": "水果摄入频率（Fruit intake）", "batch_column": "水果摄入频率(Fruit intake)", "kind": "category", "aliases": ["水果摄入频率", "水果", "Fruit"],
                    "options": [option("很少/从不", 1, "很少", "从不"), option("偶尔", 2, "偶尔"),
                               option("经常", 3, "经常"), option("每天", 4, "每天")]},
    "乳制品摄入频率": {"label": "乳制品摄入频率（Dairy intake）", "batch_column": "乳制品摄入频率(Dairy intake)", "kind": "category", "aliases": ["乳制品摄入频率", "乳制品", "Dairy"],
                      "options": [option("很少/从不", 1, "很少", "从不"), option("偶尔", 2, "偶尔"),
                                 option("经常", 3, "经常"), option("每天", 4, "每天")]},
    "高脂高糖加工食品": {"label": "高脂高糖加工食品（High-fat/sugar food）", "batch_column": "高脂高糖加工食品(High-fat food)", "kind": "category", "aliases": ["高脂高糖加工食品", "高脂高糖", "加工食品"],
                       "options": [option("否", 1, "否", "无", "很少"), option("是", 2, "是", "有", "经常")]},
    # 中医体质
    "平和质得分": {"label": "平和质得分（Gentleness score）", "batch_column": "平和质得分", "kind": "numeric", "aliases": ["平和质", "平和质得分"]},
    "气虚质得分": {"label": "气虚质得分（Qi deficiency score）", "batch_column": "气虚质得分", "kind": "numeric", "aliases": ["气虚质", "气虚质得分"]},
    "阳虚质得分": {"label": "阳虚质得分（Yang deficiency score）", "batch_column": "阳虚质得分", "kind": "numeric", "aliases": ["阳虚质", "阳虚质得分"]},
    "阴虚质得分": {"label": "阴虚质得分（Yin deficiency score）", "batch_column": "阴虚质得分", "kind": "numeric", "aliases": ["阴虚质", "阴虚质得分"]},
    "痰湿质得分": {"label": "痰湿质得分（Phlegm-dampness score）", "batch_column": "痰湿质得分", "kind": "numeric", "aliases": ["痰湿质", "痰湿质得分"]},
    "湿热质得分": {"label": "湿热质得分（Damp-heat score）", "batch_column": "湿热质得分", "kind": "numeric", "aliases": ["湿热质", "湿热质得分"]},
    "血瘀质得分": {"label": "血瘀质得分（Blood stasis score）", "batch_column": "血瘀质得分", "kind": "numeric", "aliases": ["血瘀质", "血瘀质得分"]},
    "气郁质得分": {"label": "气郁质得分（Qi stagnation score）", "batch_column": "气郁质得分", "kind": "numeric", "aliases": ["气郁质", "气郁质得分"]},
    "特禀质得分": {"label": "特禀质得分（Special constitution score）", "batch_column": "特禀质得分", "kind": "numeric", "aliases": ["特禀质", "特禀质得分"]},
    # 实验室检查
    "WBC": {"label": "白细胞计数 WBC（10⁹/L）", "batch_column": "WBC", "kind": "numeric", "aliases": ["WBC", "白细胞"]},
    "PLT": {"label": "血小板计数 PLT（10⁹/L）", "batch_column": "PLT", "kind": "numeric", "aliases": ["PLT", "血小板"]},
    "MPV": {"label": "平均血小板体积 MPV（fL）", "batch_column": "MPV", "kind": "numeric", "aliases": ["MPV", "平均血小板体积"]},
    "RDW": {"label": "红细胞分布宽度 RDW（%）", "batch_column": "RDW", "kind": "numeric", "aliases": ["RDW", "红细胞分布宽度"]},
    "NLR": {"label": "中性粒细胞/淋巴细胞比值 NLR", "batch_column": "NLR", "kind": "numeric", "aliases": ["NLR"]},
    "MLR": {"label": "单核细胞/淋巴细胞比值 MLR", "batch_column": "MLR", "kind": "numeric", "aliases": ["MLR"]},
    "PLR": {"label": "血小板/淋巴细胞比值 PLR", "batch_column": "PLR", "kind": "numeric", "aliases": ["PLR"]},
    "PNR": {"label": "血小板/中性粒细胞比值 PNR", "batch_column": "PNR", "kind": "numeric", "aliases": ["PNR"]},
    "pH": {"label": "尿液pH", "batch_column": "尿pH", "kind": "category", "aliases": ["pH", "尿pH", "尿液pH"],
           "options": [option(str(v), v, str(v)) for v in [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]]},
    "尿比重": {"label": "尿比重（Urine specific gravity）", "batch_column": "尿比重", "kind": "numeric", "aliases": ["尿比重", "比重"]},
    "尿蛋白": {"label": "尿蛋白（Urine protein）", "batch_column": "尿蛋白", "kind": "category", "aliases": ["尿蛋白", "蛋白"],
              "options": [option("阴性", 1, "阴性", "无", "-"), option("阳性", 2, "阳性", "有", "+")]},
    "尿潜血": {"label": "尿潜血（Urine occult blood）", "batch_column": "尿潜血", "kind": "category", "aliases": ["尿潜血", "潜血"],
              "options": [option("阴性", 1, "阴性", "无", "-"), option("阳性", 2, "阳性", "有", "+")]},
}

# 脉象参数自动生成specs
def add_pulse_specs():
    for side in ["左", "右"]:
        for param in ["h1", "h2", "h3", "h4", "h5", "h3/h1", "h4/h1", "h5/h1",
                       "t", "t1", "t2", "t3", "t4", "t5", "t1/t", "t4/t5",
                       "w1", "w2", "w1/t", "a_s", "a_d"]:
            key = f"{side}{param}"
            FEATURE_SPECS[key] = {
                "label": f"{side}脉 {param}",
                "batch_column": key,
                "kind": "numeric",
                "aliases": [key],
            }
        for cat, opts in [("脉力", PULSE_FORCE_OPTIONS), ("脉位", PULSE_POSITION_OPTIONS),
                           ("脉势", PULSE_QUALITY_OPTIONS), ("脉律", PULSE_RHYTHM_OPTIONS),
                           ("脉率", PULSE_RATE_OPTIONS), ("脉名", PULSE_NAME_OPTIONS)]:
            key = f"{side}{cat}"
            FEATURE_SPECS[key] = {
                "label": f"{side}脉{cat}（{side} pulse {cat}）",
                "batch_column": key,
                "kind": "category",
                "aliases": [key],
                "options": opts,
            }

add_pulse_specs()


# =============================================================================
# 3. 模型包读取与校验
# =============================================================================
class CMMInputError(ValueError):
    """用于向网页用户展示可修改的数据问题。"""


@st.cache_resource(show_spinner=False)
def load_model_bundle(bundle_path: str, modified_time: float) -> dict[str, Any]:
    del modified_time
    with Path(bundle_path).open("rb") as file:
        bundle = pickle.load(file)

    required_keys = {"model", "model_features", "preprocessor", "threshold", "positive_class", "negative_class"}
    missing_keys = sorted(required_keys.difference(bundle))
    if missing_keys:
        raise KeyError(f"模型包缺少必要内容：{missing_keys}")

    features = [str(item) for item in bundle["model_features"]]
    if not features:
        raise ValueError("模型特征为空。")

    threshold = float(bundle["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("模型包中的预测阈值必须位于0与1之间。")

    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise TypeError("模型对象不支持 predict_proba。")

    bundle["model_features"] = features
    bundle["threshold"] = threshold
    return bundle


# =============================================================================
# 4. 输入编码与预测
# =============================================================================
def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\ufeff", "").replace("\u200b", "").strip().lower()
    text = text.replace("（", "(").replace("）", ")").replace("：", ":")
    return re.sub(r"\s+", "", text)


def is_missing_input_value(value: Any) -> bool:
    if pd.isna(value):
        return True
    if not isinstance(value, str):
        return False
    return normalize_text(value) in {"", "na", "n/a", "nan", "null", "none", "#n/a"}


def display_name(feature: str) -> str:
    spec = FEATURE_SPECS.get(feature)
    return spec["label"] if spec else feature


def batch_column_name(feature: str) -> str:
    spec = FEATURE_SPECS.get(feature)
    return spec["batch_column"] if spec else feature


def feature_candidates(feature: str) -> list[str]:
    spec = FEATURE_SPECS.get(feature, {})
    candidates = [feature, batch_column_name(feature), spec.get("label", "")]
    candidates.extend(spec.get("aliases", []))
    return [c for c in candidates if c]


def resolve_input_column(data: pd.DataFrame, feature: str) -> str | None:
    if feature in data.columns:
        return feature
    normalized_columns: dict[str, str] = {}
    for column in data.columns:
        normalized_columns.setdefault(normalize_text(column), column)
    for candidate in feature_candidates(feature):
        matched = normalized_columns.get(normalize_text(candidate))
        if matched is not None:
            return matched
    return None


def resolve_named_column(data: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns: dict[str, str] = {}
    for column in data.columns:
        normalized_columns.setdefault(normalize_text(column), column)
    for candidate in candidates:
        matched = normalized_columns.get(normalize_text(candidate))
        if matched is not None:
            return matched
    return None


def encode_category_value(value: Any, feature: str, row_number: int, allow_missing: bool = False) -> Any:
    spec = FEATURE_SPECS[feature]
    codes = [item["code"] for item in spec["options"]]

    if is_missing_input_value(value):
        if allow_missing:
            return np.nan
        raise CMMInputError(f"第 {row_number} 行的“{display_name(feature)}”为空。")

    # 直接匹配code
    for code in codes:
        if value == code:
            return code

    # 匹配label或alias
    normalized_value = normalize_text(value)
    for item in spec["options"]:
        for accepted in [item["label"], *item["aliases"]]:
            if normalize_text(accepted) == normalized_value:
                return item["code"]

    # 尝试数字匹配
    try:
        numeric_value = float(normalized_value)
        for code in codes:
            try:
                if float(code) == numeric_value:
                    return code
            except (ValueError, TypeError):
                continue
    except ValueError:
        pass

    allowed_labels = "、".join(item["label"] for item in spec["options"])
    raise CMMInputError(f"第 {row_number} 行的“{display_name(feature)}”取值为“{value}”，允许：{allowed_labels}")


def encode_numeric_value(value: Any, feature: str, row_number: int, allow_missing: bool = False) -> float:
    if is_missing_input_value(value):
        if allow_missing:
            return np.nan
        raise CMMInputError(f"第 {row_number} 行的“{display_name(feature)}”为空。")
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or not np.isfinite(float(numeric_value)):
        raise CMMInputError(f"第 {row_number} 行的“{display_name(feature)}”必须为数字；当前为“{value}”。")
    return float(numeric_value)


def prepare_model_frame(raw_data: pd.DataFrame, bundle: dict[str, Any], allow_missing: bool = False) -> pd.DataFrame:
    if raw_data.empty:
        raise CMMInputError("上传文件没有数据行。")
    if raw_data.columns.duplicated().any():
        duplicated = raw_data.columns[raw_data.columns.duplicated()].astype(str).tolist()
        raise CMMInputError(f"上传文件存在重复列名：{duplicated}")

    model_features = bundle["model_features"]
    preprocessor = bundle["preprocessor"]
    le_dict = preprocessor.get("label_encoders", {})
    fill_values = preprocessor.get("fill_values", {})
    scaler = preprocessor["scaler"]

    # 构建编码后的数据
    object_cols = preprocessor.get("object_cols", [])
    encoded = pd.DataFrame(index=raw_data.index)
    for feature in model_features:
        source_col = resolve_input_column(raw_data, feature)
        if source_col is None:
            if allow_missing:
                encoded[feature] = np.nan
                continue
            raise CMMInputError(f"上传文件缺少模型需要的列：{display_name(feature)}（{batch_column_name(feature)}）")

        spec = FEATURE_SPECS.get(feature)
        values = []
        for position, value in enumerate(raw_data[source_col].tolist(), start=2):
            if spec and spec["kind"] == "category":
                values.append(encode_category_value(value, feature, position, allow_missing))
            else:
                values.append(encode_numeric_value(value, feature, position, allow_missing))
        encoded[feature] = values

    # 字符串分类变量用LabelEncoder编码（脉力、脉位、脉势、脉律、脉率、脉名等）
    for col in object_cols:
        if col in encoded.columns and col in le_dict:
            le = le_dict[col]
            encoded[col] = encoded[col].apply(
                lambda x: le.transform([str(x).strip()])[0]
                if pd.notna(x) and str(x).strip() in le.classes_ else np.nan
            )

    # 强制数值转换
    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

    # 缺失值填充
    if allow_missing:
        missing_before = int(encoded.isna().sum().sum())
        for col in encoded.columns:
            if encoded[col].isna().any():
                if col in fill_values:
                    encoded[col] = encoded[col].fillna(fill_values[col])
                else:
                    col_mean = encoded[col].mean()
                    encoded[col] = encoded[col].fillna(col_mean if pd.notna(col_mean) else 0)
        encoded.attrs["batch_missing_count"] = missing_before
        encoded.attrs["batch_missing_filled_count"] = missing_before - int(encoded.isna().sum().sum())
    elif encoded.isna().any().any():
        raise CMMInputError("编码后的模型输入仍含缺失值，请检查原始数据。")

    # 标准化
    encoded = encoded.loc[:, model_features].astype(float)
    X_scaled = scaler.transform(encoded)
    return pd.DataFrame(X_scaled, columns=model_features, index=encoded.index)


def predict_risk(model_input: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    model = bundle["model"]
    probabilities = np.asarray(model.predict_proba(model_input), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(model_input):
        raise ValueError("模型返回的概率矩阵形状异常。")

    high_class = bundle["positive_class"]
    low_class = bundle["negative_class"]
    classes = list(getattr(model, "classes_", [0, 1]))
    high_index = classes.index(high_class) if high_class in classes else 1
    probability_high = probabilities[:, high_index]
    probability_low = 1.0 - probability_high

    threshold = float(bundle["threshold"])
    predicted_label_cn = np.where(probability_high >= threshold, "高风险", "低风险")
    predicted_label_en = np.where(probability_high >= threshold, "High risk", "Low risk")

    return pd.DataFrame(
        {
            "风险等级(Risk level)": predicted_label_cn,
            "英文结果(English result)": predicted_label_en,
            "高风险概率(Probability of high risk)": probability_high,
            "低风险概率(Probability of low risk)": probability_low,
            "判定阈值(Decision threshold)": np.full(len(model_input), threshold, dtype=float),
        },
        index=model_input.index,
    )


def read_uploaded_table(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file, sheet_name=0)
    if suffix == ".csv":
        raw_bytes = uploaded_file.getvalue()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise CMMInputError("CSV编码无法识别，请另存为UTF-8或GB18030。")
    raise CMMInputError("仅支持 .xlsx 和 .csv 文件。")


# =============================================================================
# 5. 批量模板与结果文件
# =============================================================================
def create_template_workbook(model_features: list[str]) -> bytes:
    input_columns = ["对象编号(ID)"]
    for feature in model_features:
        input_columns.append(batch_column_name(feature))
    input_template = pd.DataFrame(columns=input_columns)

    instructions = pd.DataFrame({
        "变量名": [display_name(f) for f in model_features],
        "模板列名": [batch_column_name(f) for f in model_features],
        "填写说明": [
            "；".join(item["label"] for item in FEATURE_SPECS[f]["options"])
            if FEATURE_SPECS.get(f, {}).get("kind") == "category"
            else "请填写数字"
            for f in model_features
        ],
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        input_template.to_excel(writer, sheet_name="Input", index=False)
        instructions.to_excel(writer, sheet_name="填写说明", index=False)
        for sheet_name in ["Input", "填写说明"]:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 50)
    return output.getvalue()


def dataframe_to_excel_bytes(data: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Prediction results", index=False)
        ws = writer.sheets["Prediction results"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            max_len = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 2, 12), 50)
    return output.getvalue()


# =============================================================================
# 6. Streamlit界面
# =============================================================================
st.set_page_config(
    page_title="CMM-SCREEN｜三高共病风险预测工具",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1420px; padding-top: 3.0rem; padding-bottom: 1.8rem;}
        .top-blue-line {height: 25px; background: #d6e8f7; border: 1px solid #b8d4ee; margin-bottom: 13px;}
        .brand {font-size: 3.6rem; line-height: 1; font-weight: 950; color: #1a5fa8;
                letter-spacing: -2px; font-family: Arial Black, Arial, sans-serif;}
        .institutional {font-size: 1.4rem; font-weight: 800; font-style: italic; color: #0d3b6e;
                        text-align: right; padding-top: 1.0rem;}
        .thin-rule {height: 13px; border: 1px solid #c9c9c9; background: #fafafa; margin: 10px 0 7px;}
        .main-title {text-align: center; color: #27303d; font-family: Georgia, 'Times New Roman', serif;
                     font-size: 1.85rem; font-weight: 750; line-height: 1.35;}
        .cn-title {text-align: center; color: #27303d; font-size: 1.25rem; font-weight: 750; margin-top: 4px;}
        .short-rule {width: 48%; height: 11px; border: 1px solid #c9c9c9; background: #fafafa;
                     margin: 7px auto 15px;}
        .section-prompt {color: #1a5fa8; font-weight: 800; font-size: 1.05rem; margin: 3px 0 10px;}
        .intro-card {border-left: 5px solid #1a5fa8; background: #f3f8fd; padding: 0.8rem 1rem;
                     color: #26332a; margin: 0.2rem 0 1rem; line-height: 1.65;}
        .result-high {border: 2px solid #c45630; background: #fff5ef; border-radius: 9px;
                      padding: 1rem 1.2rem; font-size: 1.08rem;}
        .result-low {border: 2px solid #2c8b25; background: #eff9ed; border-radius: 9px;
                     padding: 1rem 1.2rem; font-size: 1.08rem;}
        .bottom-note {font-size: 0.88rem; line-height: 1.45; color: #303030; margin-top: 1.2rem;}
        .note-label {color: #a21414; font-weight: 800;}
        .question-label {font-size: 0.9rem; font-weight: 650; line-height: 1.4;
                         color: #333a46; padding: 0 0 0.2rem; margin-top: 0.2rem;}
        div[data-testid="stForm"] {border: 1px solid #d4d7db; border-radius: 11px;
                                   padding: 1.25rem 1.35rem 0.9rem; background: #ffffff;}
        div[data-testid="stRadio"] {padding: 0 0 0.6rem; margin: 0;}
        div[data-testid="stRadio"] > label {display: none;}
        div[data-testid="stRadio"] div[role="radiogroup"] {gap: 0.25rem 0.8rem; flex-wrap: wrap;}
        div[data-testid="stRadio"] label p {font-size: 0.85rem; line-height: 1.3; white-space: normal;}
        div[data-testid="stNumberInput"] {padding: 0 0 0.6rem; margin: 0;}
        div[data-testid="stNumberInput"] > label {display: none;}
        div[data-testid="stNumberInput"] input {min-height: 2.4rem; height: 2.4rem; font-size: 0.88rem;}
        div[data-testid="stFormSubmitButton"] {text-align: center;}
        div[data-testid="stFormSubmitButton"] button {min-width: 220px; height: 52px; font-size: 1.05rem;
                                                       font-weight: 800; border: 1.5px solid #777;}
        .stTabs [data-baseweb="tab-list"] {gap: 0.6rem;}
        .stTabs [data-baseweb="tab"] {font-weight: 700;}
        .group-subheader {font-size: 1.0rem; font-weight: 700; color: #1a5fa8;
                          border-bottom: 2px solid #d6e8f7; padding-bottom: 4px; margin: 0.8rem 0 0.6rem;}
        @media (max-width: 760px) {
            .brand {font-size: 2.2rem;}
            .institutional {font-size: 1.0rem;}
            .main-title {font-size: 1.1rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    brand_col, use_col = st.columns([2, 1])
    with brand_col:
        st.markdown('<div class="brand">CMM-SCREEN</div>', unsafe_allow_html=True)
    with use_col:
        st.markdown('<div class="institutional">For Institutional Use</div>', unsafe_allow_html=True)
    st.markdown('<div class="thin-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-title">Web-based tool for predicting cardiometabolic multimorbidity risk '
        'among community residents</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cn-title">社区居民三高共病风险预测工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="short-rule"></div>', unsafe_allow_html=True)


def render_feature_widget(feature: str) -> Any:
    spec = FEATURE_SPECS.get(feature)
    label = display_name(feature)
    st.markdown(f'<div class="question-label">{label}</div>', unsafe_allow_html=True)

    if spec is None:
        return st.number_input(label, value=0.0, step=1.0, key=f"single_{feature}", label_visibility="collapsed")

    if spec["kind"] == "category":
        labels = [item["label"] for item in spec["options"]]
        return st.radio(label, labels, horizontal=True, key=f"single_{feature}", label_visibility="collapsed")

    return st.number_input(label, value=0.0, step=0.1, key=f"single_{feature}",
                           help="请按实际检查结果填写数字", label_visibility="collapsed")


def render_single_prediction(bundle: dict[str, Any]) -> None:
    st.markdown(
        '<div class="section-prompt">Please submit the following information / 请输入以下信息：</div>',
        unsafe_allow_html=True,
    )
    features = bundle["model_features"]

    with st.form("single_prediction_form", clear_on_submit=False):
        submitted_values: dict[str, Any] = {}

        # 基本信息 + 生活方式
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="group-subheader">👤 基本信息</div>', unsafe_allow_html=True)
            for f in FEATURE_GROUPS["基本信息"]:
                if f in features:
                    submitted_values[f] = render_feature_widget(f)
        with col2:
            st.markdown('<div class="group-subheader">🍽️ 生活方式</div>', unsafe_allow_html=True)
            for f in FEATURE_GROUPS["生活方式"]:
                if f in features:
                    submitted_values[f] = render_feature_widget(f)

        # 中医体质 + 实验室检查
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="group-subheader">🌿 中医体质（0-100分）</div>', unsafe_allow_html=True)
            for f in FEATURE_GROUPS["中医体质"]:
                if f in features:
                    submitted_values[f] = render_feature_widget(f)
        with col2:
            st.markdown('<div class="group-subheader">🔬 实验室检查</div>', unsafe_allow_html=True)
            for f in FEATURE_GROUPS["实验室检查"]:
                if f in features:
                    submitted_values[f] = render_feature_widget(f)

        # 脉象参数（折叠）
        with st.expander("🩺 脉象参数（可选，留空将用默认值）", expanded=False):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<div class="group-subheader">左脉参数</div>', unsafe_allow_html=True)
                for f in FEATURE_GROUPS["左脉参数"]:
                    if f in features:
                        submitted_values[f] = render_feature_widget(f)
            with col2:
                st.markdown('<div class="group-subheader">右脉参数</div>', unsafe_allow_html=True)
                for f in FEATURE_GROUPS["右脉参数"]:
                    if f in features:
                        submitted_values[f] = render_feature_widget(f)

        submitted = st.form_submit_button("Calculate / 计算", use_container_width=False)

    if not submitted:
        return

    try:
        # 为折叠面板中未创建的特征填充默认值
        object_cols = bundle["preprocessor"].get("object_cols", [])
        for f in features:
            if f not in submitted_values:
                spec = FEATURE_SPECS.get(f)
                if spec and spec["kind"] == "category":
                    submitted_values[f] = spec["options"][0]["code"]
                else:
                    submitted_values[f] = 0.0

        raw_input = pd.DataFrame([submitted_values])
        model_input = prepare_model_frame(raw_input, bundle)
        result = predict_risk(model_input, bundle).iloc[0]
        prob_high = float(result["高风险概率(Probability of high risk)"])
        prob_low = float(result["低风险概率(Probability of low risk)"])
        is_high = result["风险等级(Risk level)"] == "高风险"

        if is_high:
            message = (
                f"<div class='result-high'><b>预测结果：高风险（High risk）</b><br>"
                f"该居民被判定为三高共病高风险，建议尽快就医进行血压、血糖、血脂全面检查，"
                f"改善生活方式并定期监测。高风险概率：<b>{prob_high:.1%}</b>；"
                f"低风险概率：{prob_low:.1%}。</div>"
            )
        else:
            message = (
                f"<div class='result-low'><b>预测结果：低风险（Low risk）</b><br>"
                f"该居民被判定为三高共病低风险，建议继续保持健康生活方式，定期体检。"
                f"高风险概率：{prob_high:.1%}；低风险概率：<b>{prob_low:.1%}</b>。</div>"
            )
        st.markdown(message, unsafe_allow_html=True)
        st.caption(
            f"判定规则：高风险概率 ≥ {float(bundle['threshold']):.4f} 时判为高风险，"
            f"低于 {float(bundle['threshold']):.4f} 时判为低风险；"
            "本结果仅用于科研和健康管理参考，不构成临床诊断或治疗建议。"
        )
    except Exception as error:
        st.error(f"无法完成预测：{error}")


def render_batch_prediction(bundle: dict[str, Any]) -> None:
    st.markdown(
        '<div class="section-prompt">For Institutional Use / 供机构批量使用</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "上传包含多名居民信息的 Excel（.xlsx）或 CSV 文件。工具会逐行生成高/低风险、"
        "高风险概率和低风险概率，并可下载完整结果。"
    )

    template_bytes = create_template_workbook(bundle["model_features"])
    st.download_button(
        "下载批量导入模板",
        data=template_bytes,
        file_name="CMM-SCREEN_batch_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded_file = st.file_uploader(
        "请上传填写后的表格 / Upload the completed table",
        type=["xlsx", "csv"],
        key="batch_file_uploader",
    )
    if uploaded_file is None:
        return

    try:
        batch_data = read_uploaded_table(uploaded_file)
        st.write(f"已读取 {len(batch_data)} 名居民、{len(batch_data.columns)} 列数据。")

        identifier_column = resolve_named_column(batch_data, BATCH_IDENTIFIER_ALIASES)
        if identifier_column is not None:
            unique_count = int(batch_data[identifier_column].nunique(dropna=True))
            st.info(f"已识别对象编号列：共 {unique_count} 个唯一编号。")

        with st.expander("数据预览（前5行） / Preview first 5 rows", expanded=True):
            st.dataframe(batch_data.head(5), use_container_width=True, hide_index=True)
    except Exception as error:
        st.error(f"文件读取失败：{error}")
        return

    if not st.button("Calculate / 批量计算", type="primary", key="batch_calculate"):
        return

    try:
        model_input = prepare_model_frame(batch_data, bundle, allow_missing=True)
        missing_count = int(model_input.attrs.get("batch_missing_count", 0))
        filled_count = int(model_input.attrs.get("batch_missing_filled_count", 0))
        if missing_count:
            st.warning(f"检测到 {missing_count} 个缺失值，其中 {filled_count} 个已用训练集填充值填补。")

        predictions = predict_risk(model_input, bundle)
        output_data = batch_data.reset_index(drop=True).copy()
        predictions = predictions.reset_index(drop=True)
        output_data = pd.concat([output_data, predictions], axis=1)

        st.success(f"批量预测完成：共 {len(output_data)} 名居民。")
        summary = predictions["风险等级(Risk level)"].value_counts()
        high_count = int(summary.get("高风险", 0))
        low_count = int(summary.get("低风险", 0))
        metric_total, metric_low, metric_high = st.columns(3)
        metric_total.metric("总人数", len(output_data))
        metric_low.metric("低风险", low_count)
        metric_high.metric("高风险", high_count)

        st.markdown("#### 风险分布")
        dist_data = pd.DataFrame(
            {"人数": [low_count, high_count]},
            index=pd.Index(["低风险", "高风险"], name="风险等级"),
        )
        st.bar_chart(dist_data, color=["#2c8b25", "#c45630"])

        st.caption(f"本批次统一使用阈值 {float(bundle['threshold']):.4f}。")

        # 展示精简结果
        display_data = pd.DataFrame(index=output_data.index)
        if identifier_column:
            display_data["对象编号(ID)"] = output_data[identifier_column]
        for col in BATCH_RESULT_DISPLAY_COLUMNS:
            if col in output_data.columns:
                display_data[col] = output_data[col]
        st.dataframe(display_data.head(10), use_container_width=True, hide_index=True)
        if len(output_data) > 10:
            st.caption("页面仅展示前10行，完整结果请下载。")

        download_data = output_data.drop(columns=["判定阈值(Decision threshold)"], errors="ignore")
        excel_result = dataframe_to_excel_bytes(download_data)
        csv_result = download_data.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        excel_col, csv_col = st.columns(2)
        with excel_col:
            st.download_button("下载Excel预测结果", data=excel_result,
                               file_name="CMM-SCREEN_prediction_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        with csv_col:
            st.download_button("下载CSV预测结果", data=csv_result,
                               file_name="CMM-SCREEN_prediction_results.csv",
                               mime="text/csv", use_container_width=True)
    except CMMInputError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"批量预测失败：{error}")


def render_about(bundle: dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="intro-card">
        <b>工具简介</b><br>
        CMM-SCREEN是一款面向社区居民的三高（高血压、高血糖、高血脂）共病风险预测工具，
        基于XGBoost机器学习模型，整合人口学特征、生活方式、中医体质、实验室检查和脉象参数，
        用于评估社区居民发生三高共病的风险。工作人员可通过单人录入或批量上传Excel、CSV文件获得预测结果，
        辅助识别高风险人群，为社区健康管理、慢病随访和针对性干预提供参考。
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 模型信息")
        st.write(f"- **模型类型**：{bundle.get('model_type', 'XGBoost')}")
        st.write(f"- **数据增强**：{bundle.get('augmentation', 'TomekLinks')}")
        st.write(f"- **特征数量**：{len(bundle['model_features'])}个")
        st.write(f"- **训练样本**：{bundle.get('n_samples_train', 'N/A')}例")
        st.write(f"- **训练AUC**：{bundle.get('train_auc', 'N/A'):.4f}" if isinstance(bundle.get('train_auc'), float) else "- **训练AUC**：N/A")
        st.write(f"- **判定阈值**：{float(bundle['threshold']):.4f}")
    with col2:
        st.markdown("#### 特征类别")
        st.write("- 👤 人口学特征（年龄、性别、BMI、学历等）")
        st.write("- 🍽️ 生活方式（吸烟、饮酒、饮食、运动等）")
        st.write("- 🌿 中医体质（九种体质得分）")
        st.write("- 🔬 实验室检查（血常规、尿常规、炎症指标等）")
        st.write("- 🩺 脉象参数（左右脉的h/t/w/a及分类特征）")


def render_footer() -> None:
    st.markdown(
        """
        <div class="bottom-note">
        <span class="note-label">Note：</span>
        CMM-SCREEN是一款采用人工智能开发的网页工具，用于预测社区居民的三高共病风险；
        该工具由上海中医药大学开发，旨在协助社区卫生服务机构识别高风险人群并开展针对性干预。
        预测结果仅供科研和健康管理参考，不构成临床诊断或治疗建议。
        <br>
        CMM-SCREEN is an AI-based web tool for predicting cardiometabolic multimorbidity risk among community residents.
        It was developed by Shanghai University of Traditional Chinese Medicine,
        to support community health services in identifying high-risk individuals for targeted intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# 7. 主程序
# =============================================================================
render_header()

if not BUNDLE_PATH.is_file():
    st.error(f"尚未找到模型包：{BUNDLE_PATH}")
    st.write("请先运行 train_model.py 训练模型，确保 model/cmm_xgboost_bundle.pkl 存在。")
    render_footer()
else:
    try:
        import pickle
        model_bundle = load_model_bundle(str(BUNDLE_PATH), BUNDLE_PATH.stat().st_mtime)
    except Exception as bundle_error:
        st.error(f"模型包加载失败：{bundle_error}")
        render_footer()
    else:
        single_tab, batch_tab, about_tab = st.tabs(
            ["单人预测 Individual", "机构批量预测 Batch", "工具简介 About"]
        )
        with single_tab:
            render_single_prediction(model_bundle)
        with batch_tab:
            render_batch_prediction(model_bundle)
        with about_tab:
            render_about(model_bundle)
        render_footer()
