client.chat.completions.create(
  model="gpt-4.1",
  temperature=0,
  seed=7,
  tools=[EMIT_ENTITIES_TOOL],  # schema above
  messages=[
    {"role":"system","content": SYSTEM_TEXT},
    {"role":"user","content": f"EXCLUDE_NAMES: {exclude_list}\nTEXT:\n{doc}"}
  ]
)

# After tool call:
entities = tool_args["entities"]
entities = [e for e in entities if normalize(e["normalized_name"]) not in exclude_set]
