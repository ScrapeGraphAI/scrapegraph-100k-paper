import json
import math
from collections import Counter
from typing import Any

import jsonschema_rs
import orjson
import sacrebleu

# Sentinel value for empty dicts to preserve them during flattening
EMPTY_DICT_SENTINEL = object()


def fulfills_schema(schema: dict, obj: dict) -> tuple[bool, str]:
    try:
        jsonschema_rs.validate(schema, obj)
        return True, "all good"
    except Exception as e:
        return False, str(e)


def json_validator(pred: str, true_schema: dict) -> dict:
    is_valid = False
    is_compliant = False
    try:
        parsed = orjson.loads(pred)
        is_valid = True
    except Exception as e:
        return {"is_valid": is_valid, "is_compliant": is_compliant, "error": str(e)}
    is_compliant, error_msg = fulfills_schema(true_schema, parsed)
    return {"is_valid": is_valid, "is_compliant": is_compliant, "error": error_msg}


def parse_json_remove_duplicates(json_str: str) -> dict:
    def _remove_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
        """Remove keys that appear more than once from JSON object pairs."""
        key_counts = Counter(k for k, _ in pairs)
        return {k: v for k, v in pairs if key_counts[k] == 1}

    try:
        return json.loads(json_str, object_pairs_hook=_remove_duplicate_keys)
    except json.JSONDecodeError:
        return {}


def _flatten_list(lst: list, parent_key: str, sep: str = ".") -> dict:
    """Flatten list items using [*] wildcard for order-independent key matching.

    Lists containing dicts have their inner keys extracted with [*] notation.
    For example: {"items": [{"name": "A"}, {"name": "B"}]} becomes
    {"items[*].name": ["A", "B"]}
    """
    items: list[tuple[str, Any]] = []
    has_dict_items = any(isinstance(item, dict) for item in lst)

    if has_dict_items:
        # Collect all keys from all dict items
        all_flattened: list[dict] = []
        for item in lst:
            if isinstance(item, dict):
                all_flattened.append(_flatten_dict(item, f"{parent_key}[*]", sep))
            elif isinstance(item, list):
                all_flattened.append(_flatten_list(item, f"{parent_key}[*]", sep))
        # Merge all flattened dicts - values become lists
        merged: dict[str, list] = {}
        for flat in all_flattened:
            for k, v in flat.items():
                merged.setdefault(k, []).append(v)
        # Convert to final format
        for k, vals in merged.items():
            items.append((k, vals))
    else:
        # No dicts in list - keep as single value
        items.append((parent_key, lst))

    return dict(items)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict with dot-separated keys.

    Handles edge cases:
    - Dots in keys are escaped with backslash to avoid collision
    - Empty dicts are preserved with a sentinel value
    - Lists containing dicts are flattened using [*] wildcard notation
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        # Escape dots in keys to avoid collision with nested key separator
        escaped_k = str(k).replace("\\", "\\\\").replace(".", "\\.")
        new_key = f"{parent_key}{sep}{escaped_k}" if parent_key else escaped_k
        if isinstance(v, dict):
            if len(v) == 0:
                # Preserve empty dicts with sentinel value
                items.append((new_key, EMPTY_DICT_SENTINEL))
            else:
                items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.extend(_flatten_list(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _compare_values(pred_val: Any, true_val: Any) -> float:
    # Handle empty dict sentinel
    if true_val is EMPTY_DICT_SENTINEL:
        return 1.0 if pred_val is EMPTY_DICT_SENTINEL else 0.0

    # Handle None/null
    if true_val is None:
        return 1.0 if pred_val is None else 0.0

    # Boolean comparison
    if isinstance(true_val, bool):
        return 1.0 if pred_val == true_val else 0.0

    # Numeric comparison (int or float, but not bool since bool is subclass of int)
    if isinstance(true_val, (int, float)) and not isinstance(true_val, bool):
        # Use math.isclose for float comparisons to handle precision issues
        if isinstance(true_val, float) or isinstance(pred_val, float):
            try:
                return (
                    1.0
                    if math.isclose(pred_val, true_val, rel_tol=1e-9, abs_tol=1e-12)
                    else 0.0
                )
            except TypeError:
                return 0.0
        return 1.0 if pred_val == true_val else 0.0

    # Array/list comparison (with Counter to preserve duplicates, order doesn't matter)
    if isinstance(true_val, list):
        if not isinstance(pred_val, list):
            return 0.0
        try:
            # Use Counter (multiset) comparison to preserve duplicates
            return 1.0 if Counter(pred_val) == Counter(true_val) else 0.0
        except TypeError:
            # For unhashable items (nested lists/dicts), use sorted comparison
            try:
                return (
                    1.0
                    if sorted(pred_val, key=str) == sorted(true_val, key=str)
                    else 0.0
                )
            except TypeError:
                return 1.0 if pred_val == true_val else 0.0

    # String comparison using BLEU
    if isinstance(true_val, str):
        if not isinstance(pred_val, str):
            pred_val = str(pred_val)
        if not true_val and not pred_val:
            return 1.0
        if not true_val or not pred_val:
            return 0.0
        bleu = sacrebleu.sentence_bleu(pred_val, [true_val])
        return bleu.score / 100.0  # Normalize to 0-1

    # Fallback: exact match
    return 1.0 if pred_val == true_val else 0.0


def magic_metric_failed() -> dict[str, float]:
    return {
        "key_precision": 0.0,
        "key_recall": 0.0,
        "key_f1": 0.0,
        "missing_keys": 0.0,
        "extra_keys": 0.0,
        "value_score": 0.0,
        "overall_bleu": 0.0,
    }


def magic_metric(pred: dict | list, true: dict | list) -> dict[str, float]:
    """
    - key_precision: fraction of pred keys that exist in true
    - key_recall: fraction of true keys that exist in pred
    - key_f1: harmonic mean of precision and recall
    - missing_keys: count of keys in true but not in pred
    - extra_keys: count of keys in pred but not in true
    - value_score: average field score (missing keys count as 0)
    - overall_bleu: BLEU score on stringified JSONs
    """
    # Normalize root-level arrays to dict with synthetic key
    if isinstance(pred, list):
        pred = {"[root]": pred}
    if isinstance(true, list):
        true = {"[root]": true}

    # Flatten both dictionaries
    flat_pred = _flatten_dict(pred)
    flat_true = _flatten_dict(true)

    pred_keys = set(flat_pred.keys())
    true_keys = set(flat_true.keys())

    # Key-level metrics
    common_keys = pred_keys & true_keys
    missing_keys = len(true_keys - pred_keys)
    extra_keys = len(pred_keys - true_keys)

    if len(pred_keys) > 0:
        key_precision = len(common_keys) / len(pred_keys)
    else:
        key_precision = 0.0 if len(true_keys) > 0 else 1.0

    if len(true_keys) > 0:
        key_recall = len(common_keys) / len(true_keys)
    else:
        key_recall = 1.0 if len(pred_keys) == 0 else 0.0

    if key_precision + key_recall > 0:
        key_f1 = 2 * key_precision * key_recall / (key_precision + key_recall)
    else:
        key_f1 = 0.0

    # Value-level metrics (score per true key, missing = 0)
    if len(true_keys) > 0:
        value_scores = []
        for key in true_keys:
            if key in flat_pred:
                score = _compare_values(flat_pred[key], flat_true[key])
            else:
                score = 0.0  # Missing key = 0
            value_scores.append(score)
        value_score = sum(value_scores) / len(value_scores)
    else:
        value_score = 1.0 if len(pred_keys) == 0 else 0.0

    pred_str = orjson.dumps(pred, option=orjson.OPT_SORT_KEYS).decode()
    true_str = orjson.dumps(true, option=orjson.OPT_SORT_KEYS).decode()
    if pred_str and true_str:
        overall_bleu = sacrebleu.sentence_bleu(pred_str, [true_str]).score / 100.0
    else:
        overall_bleu = 1.0 if pred_str == true_str else 0.0

    return {
        "key_precision": key_precision,
        "key_recall": key_recall,
        "key_f1": key_f1,
        "missing_keys": float(missing_keys),
        "extra_keys": float(extra_keys),
        "value_score": value_score,
        "overall_bleu": overall_bleu,
    }


def run_all(pred_str: str, schema: dict, ground_truth: dict | list) -> dict[str, float]:
    validation = json_validator(pred_str, schema)
    result = {
        "is_valid_json": float(validation["is_valid"]),
        "is_schema_compliant": float(validation["is_compliant"]),
    }
    if validation["is_valid"]:
        pred = parse_json_remove_duplicates(pred_str)
    else:
        pred = {} if isinstance(ground_truth, dict) else []
    result.update(magic_metric(pred, ground_truth))
    return result
