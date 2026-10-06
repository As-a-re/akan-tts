"""Akan (Twi) text normalisation for TTS.

Design choices (discussed in the report):
  * Twi orthography is close to phonemic, so we use GRAPHEMES as model input
    (no G2P). Open vowels are kept as the distinct symbols  ɛ  and  ɔ.
  * Standard Twi orthography does not mark tone, so tone diacritics are stripped.
  * Digits 0-99 are expanded to Twi number words (VERIFY with a native speaker).
"""
import re
import unicodedata

# Characters that are visually/keyboard variants of the Akan letters
CHAR_MAP = {
    "ε": "ɛ", "Ε": "ɛ", "Ɛ": "ɛ", "ϵ": "ɛ", "є": "ɛ",     # Greek/Cyrillic look-alikes + capital
    "Ɔ": "ɔ", "ᴐ": "ɔ",
    "\u2019": "'", "\u2018": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "«": '"', "»": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}

LETTERS = "abcdefghijklmnopqrstuvwxyzɛɔ"
PUNCT = " ,.?!-'\""
ALLOWED = set(LETTERS + PUNCT)

UNITS = {0: "hwee", 1: "baako", 2: "mmienu", 3: "mmiɛnsa", 4: "nnan", 5: "nnum",
         6: "nsia", 7: "nson", 8: "nwɔtwe", 9: "nkron"}
TEENS = {10: "edu", 11: "dubaako", 12: "dumienu", 13: "dumiɛnsa", 14: "duanan",
         15: "duanum", 16: "duosia", 17: "duoson", 18: "duowɔtwe", 19: "duokron"}
TENS = {2: "aduonu", 3: "aduasa", 4: "aduanan", 5: "aduonum",
        6: "aduosia", 7: "aduoson", 8: "aduowɔtwe", 9: "aduokron"}


def number_to_twi(n: int) -> str:
    """0-99 -> Twi words. Larger numbers are read digit by digit (documented limitation)."""
    if 0 <= n <= 9:
        return UNITS[n]
    if 10 <= n <= 19:
        return TEENS[n]
    if 20 <= n <= 99:
        t, u = divmod(n, 10)
        return TENS[t] if u == 0 else f"{TENS[t]} {UNITS[u]}"
    return " ".join(UNITS[int(d)] for d in str(n))


def _expand_numbers(text: str) -> str:
    return re.sub(r"\d+", lambda m: " " + number_to_twi(int(m.group())) + " ", text)


def strip_tone_marks(text: str) -> str:
    """Remove combining diacritics (tone marks). ɛ and ɔ have no canonical decomposition, so they survive."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in nfd if not unicodedata.combining(c)))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(CHAR_MAP.get(c, c) for c in text)
    text = strip_tone_marks(text)
    text = text.lower()
    text = _expand_numbers(text)
    text = "".join(c for c in text if c in ALLOWED)       # drop everything else
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.?!])", r"\1", text)      # no space before punctuation
    return text


def is_valid(text: str, min_chars: int = 3) -> bool:
    return len(re.sub(r"[^a-zɛɔ]", "", text)) >= min_chars


if __name__ == "__main__":
    for s in ["Ɛyɛ fɛ paa!", "Mewɔ mfe 25.", "Me pɛ sɛ meɔkɔ sukuu", "Εnnɛ ewiem yɛ hyew."]:
        print(s, "->", normalize_text(s))
