def build(schema: str, content: str) -> str:
    return f"""Extract data from the content according to the JSON schema.
Schema: {schema}
Content: {content}
Return ONLY valid JSON matching the schema."""
