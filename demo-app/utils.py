"""Utility functions — deliberately contains 3 bugs for the QA agent to find and fix."""


def safe_divide(a: float, b: float) -> float:
    """Divide a by b. Returns 0.0 if b is zero."""
    # BUG 1: missing zero-guard — raises ZeroDivisionError instead of returning 0.0
    return a / b


def is_palindrome(text: str) -> bool:
    """Return True if text reads the same forwards and backwards (case-insensitive)."""
    # BUG 2: case-sensitive comparison — "Racecar" wrongly returns False
    return text == text[::-1]


def word_count(text: str) -> int:
    """Count words in a string, ignoring extra whitespace."""
    # BUG 3: split(" ") fails on tabs and multiple consecutive spaces
    words = text.split(" ")
    return len([w for w in words if w])


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9
