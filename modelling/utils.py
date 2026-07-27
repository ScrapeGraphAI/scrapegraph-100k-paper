import re

RE_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def extract_json(text: str) -> str:
    if match := RE_JSON_BLOCK.search(text):
        return match.group(1).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text

    first_idx = min(i for i in (first_brace, first_bracket) if i != -1)
    open_char = text[first_idx]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    escape_next = False

    for i in range(first_idx, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[first_idx : i + 1]

    return text


if __name__ == "__main__":
    import json

    tests = [
        ('{"foo": 1}', '{"foo": 1}'),
        ('```json\n{"bar": 2}\n```', '{"bar": 2}'),
        ('```\n{"baz": 3}\n```', '{"baz": 3}'),
        ('Here is your JSON:\n{"x": 10}\nHope that helps!', '{"x": 10}'),
        (
            'Sure! ```json\n{"nested": {"a": 1}}\n``` Let me know.',
            '{"nested": {"a": 1}}',
        ),
        ('blah blah {"key": "value"} more blah', '{"key": "value"}'),
        ('  \n{"with_whitespace": true}\n  ', '{"with_whitespace": true}'),
        ('{"foo":1}', '{"foo": 1}'),
        (
            '{"outer": {"inner": {"deep": [1, 2, 3]}}}',
            '{"outer": {"inner": {"deep": [1, 2, 3]}}}',
        ),
        ('Here: {"a": {"b": {"c": "nested"}}} done', '{"a": {"b": {"c": "nested"}}}'),
        ('["a", "b", "c"]', '["a", "b", "c"]'),
        ("```json\n[1, 2, 3]\n```", "[1, 2, 3]"),
        ('Here is the array: [{"x": 1}, {"y": 2}] thanks!', '[{"x": 1}, {"y": 2}]'),
        ('[{"nested": {"deep": "value"}}]', '[{"nested": {"deep": "value"}}]'),
    ]

    failed = 0
    for text, expected in tests:
        try:
            result = extract_json(text)
            assert json.loads(result) == json.loads(
                expected
            ), f"Expected {expected}, got {result}"
            print(f"✓ {text[:40]!r}...")
        except Exception as e:
            print(f"✗ {text[:40]!r}... -> {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
