def system_prompt(content: str) -> dict[str, str]:
    return {"role": "system",
            "content": content}


def user_prompt(content: str) -> dict[str, str]:
    return {"role": "user",
            "content": content}


def assistant_prompt(content: str) -> dict[str, str]:
    return {"role": "assistant",
            "content": content}