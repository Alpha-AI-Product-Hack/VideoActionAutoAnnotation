PROMPT_ID = "the_action_is"
PROMPT_TEMPLATE = "the action is: {action_label}"
LABEL_ONLY_PROMPT_ID = "label_only"


def render_action_prompt(action_label: str, prompt_id: str = PROMPT_ID) -> str:
    if prompt_id == PROMPT_ID:
        return PROMPT_TEMPLATE.format(action_label=action_label)
    if prompt_id == LABEL_ONLY_PROMPT_ID:
        return action_label
    raise ValueError(f"Unsupported prompt_id={prompt_id!r}")
