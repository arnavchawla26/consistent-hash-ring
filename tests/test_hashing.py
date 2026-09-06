from chring.hashing import MASK_64, hash_to_int


def test_deterministic():
    assert hash_to_int("hello") == hash_to_int("hello")


def test_different_inputs_differ():
    assert hash_to_int("hello") != hash_to_int("world")


def test_within_64_bits():
    for s in ["", "a", "a much longer string of text than the others", "\U0001f389unicode"]:
        h = hash_to_int(s)
        assert 0 <= h <= MASK_64


def test_empty_string_hashes():
    # should not raise
    hash_to_int("")
