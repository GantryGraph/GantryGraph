"""Test suite for utils.py — several tests will FAIL due to known bugs."""
import pytest
from utils import (
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    is_palindrome,
    safe_divide,
    word_count,
)


# ── safe_divide ───────────────────────────────────────────────────────────────

def test_safe_divide_normal():
    assert safe_divide(10.0, 2.0) == 5.0


def test_safe_divide_negative():
    assert safe_divide(-9.0, 3.0) == -3.0


def test_safe_divide_by_zero():
    # BUG: currently raises ZeroDivisionError instead of returning 0.0
    assert safe_divide(7.0, 0.0) == 0.0


# ── is_palindrome ─────────────────────────────────────────────────────────────

def test_is_palindrome_lowercase():
    assert is_palindrome("racecar") is True


def test_is_palindrome_not_palindrome():
    assert is_palindrome("hello") is False


def test_is_palindrome_case_insensitive():
    # BUG: currently returns False because 'R' != 'r'
    assert is_palindrome("Racecar") is True


def test_is_palindrome_mixed_case():
    assert is_palindrome("abcba") is True
    assert is_palindrome("xyz") is False


# ── word_count ────────────────────────────────────────────────────────────────

def test_word_count_simple():
    assert word_count("hello world") == 2


def test_word_count_extra_spaces():
    # BUG: split(" ") on double space gives empty string in list
    assert word_count("hello  world") == 2


def test_word_count_tabs():
    # BUG: split(" ") doesn't split on tabs
    assert word_count("hello\tworld") == 2


def test_word_count_empty():
    assert word_count("") == 0


# ── Temperature conversions (these PASS — no bugs) ───────────────────────────

def test_celsius_to_fahrenheit_boiling():
    assert celsius_to_fahrenheit(100) == pytest.approx(212.0)


def test_celsius_to_fahrenheit_freezing():
    assert celsius_to_fahrenheit(0) == pytest.approx(32.0)


def test_fahrenheit_to_celsius_body_temp():
    assert fahrenheit_to_celsius(98.6) == pytest.approx(37.0, abs=0.1)
