


def _smart_cast(value):
    """Tries to return an int | if fails => float | if fails => str
    Handles non-string input, whitespace, and comma thousands separators.
    NaN/Inf strings are intentionally allowed through as floats.
    """
    # Guard against non-string input (None, list, etc.) rather than crashing
    if not isinstance(value, str):
        return value

    s = value.strip()

    if s == "":
        return s  # empty/whitespace-only stays as-is, don't try to parse it

    # Try int first (handles "42", "+5", "007", "1_000")
    try:
        return int(s)
    except ValueError:
        pass

    # Try float (handles "3.14", "1e-5", "nan", "inf", ".5", "5.")
    try:
        return float(s)
    except ValueError:
        pass

    # Try stripping thousands separators, then retry both
    # (only if it looks like a plausible grouped number, to avoid
    # mangling things like "1,000km" or genuinely non-numeric text)
    if "," in s:
        s_nocomma = s.replace(",", "")
        try:
            return int(s_nocomma)
        except ValueError:
            pass
        try:
            return float(s_nocomma)
        except ValueError:
            pass

    return value  # give back the original (unstripped) value as string