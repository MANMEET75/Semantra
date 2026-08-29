from semantra.preprocessing import tokenize


def test_tokenize_supports_unicode_and_hinglish():
    assert tokenize("Mera account login nahi ho raha") == [
        "mera",
        "account",
        "login",
        "nahi",
        "ho",
        "raha",
    ]
    assert "मेरा" in tokenize("मेरा अकाउंट लॉगिन नहीं हो रहा")
