# Frontier structured-output baselines vs fine-tuned model, by schema complexity

gpt-5.6 and claude-sonnet-4.6 use native structured-output mode (json_schema);
on schema rejection they fall back to json_object (rate in 'rej'). sgai-qwen3-1.7b
is plain generation (no constrained decoding), temperature 0, repetition_penalty 1.1.
claude n=2295/2761 (proxy budget exhausted; missing rows excluded).

| bucket | gpt-5.6-terra-fiit n / compliant / key-F1 / rej | claude-sonnet-4.6-fiit n / compliant / key-F1 / rej | sgai-qwen3-1.7b n / compliant / key-F1 / rej |
|---|---|---|---|
| overall | 2761 / 0.988 / 0.929 / 0.00 | 2295 / 0.984 / 0.916 / 0.62 | 2761 / 0.919 / 0.887 / 0.00 |
| 0-50 | 493 / 0.996 / 0.977 / 0.00 | 415 / 0.990 / 0.971 / 0.57 | 493 / 0.955 / 0.937 / 0.00 |
| 50-100 | 905 / 0.992 / 0.938 / 0.00 | 749 / 0.984 / 0.928 / 0.59 | 905 / 0.950 / 0.907 / 0.00 |
| 100-200 | 892 / 0.991 / 0.919 / 0.00 | 733 / 0.990 / 0.904 / 0.58 | 892 / 0.910 / 0.872 / 0.00 |
| 200-500 | 347 / 0.977 / 0.905 / 0.00 | 296 / 0.980 / 0.871 / 0.71 | 347 / 0.865 / 0.841 / 0.00 |
| 500-1000 | 92 / 0.957 / 0.861 / 0.00 | 75 / 0.893 / 0.861 / 0.96 | 92 / 0.783 / 0.826 / 0.00 |
| 1000+ | 32 / 0.844 / 0.650 / 0.00 | 27 / 1.000 / 0.706 / 0.93 | 32 / 0.719 / 0.625 / 0.00 |
| depth>=7 | 408 / 0.980 / 0.841 / 0.00 | 343 / 0.968 / 0.817 / 0.77 | 408 / 0.853 / 0.795 / 0.00 |
| keys>=200 | 72 / 0.903 / 0.732 / 0.00 | 62 / 0.919 / 0.774 / 0.97 | 72 / 0.722 / 0.708 / 0.00 |
