from app.services.passwords import hash_password, verify_password


def test_hash_is_argon2_and_not_the_password():
    hashed = hash_password("correct horse")

    assert hashed.startswith("$argon2")
    assert "correct horse" not in hashed


def test_verify_accepts_right_password_and_rejects_wrong_one():
    hashed = hash_password("correct horse")

    assert verify_password("correct horse", hashed)
    assert not verify_password("wrong horse", hashed)
