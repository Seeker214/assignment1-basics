
def build_initial_vocab(special_tokens: list[str]) -> list[bytes]:
    vocab = []
    for i in range(256):
        vocab.append(bytes([i]))
    for token in special_tokens:
        vocab.append(token.encode("utf-8"))
    return vocab