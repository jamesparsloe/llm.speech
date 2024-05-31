def tokenize(text: str):
    token_ids = list(text.encode("utf-8"))

    return token_ids


def detokenize(
    token_ids: list[int],
):
    text = "".join(chr(token_id) for token_id in token_ids)

    return text


if __name__ == "__main__":
    text = "Does this work?"
    token_ids = tokenize(text)

    print(token_ids)

    recons = detokenize(token_ids)
    print(recons)
    assert recons == text
