"""Tests for credentials_store.py.

These tests NEVER touch the real trade_data/credentials.json. Every test
points credentials_store at a throwaway path under tmp_path and uses only
fake placeholder key material.
"""
import pytest

import credentials_store as cs

FAKE_KEY = "test_key_1234567890"
FAKE_SECRET = "test_secret_abcdefghij"


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirect credentials_store's paths at a throwaway file."""
    monkeypatch.setattr(cs, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    return tmp_path


def test_masked_empty_value_returns_empty_string():
    assert cs.masked("") == ""
    assert cs.masked(None) == ""


def test_masked_short_value_is_fully_starred():
    # len <= keep*2 (default keep=4) -> fully masked, nothing leaked
    assert cs.masked("abcd") == "****"
    assert cs.masked("abcdefgh") == "********"


def test_masked_long_value_keeps_head_and_tail_only():
    result = cs.masked(FAKE_KEY, keep=4)
    assert result.startswith(FAKE_KEY[:4])
    assert result.endswith(FAKE_KEY[-4:])
    assert FAKE_KEY not in result  # the full secret is never present
    assert result.count("*") == len(FAKE_KEY) - 8


def test_load_credentials_missing_file_returns_defaults(isolated_store):
    creds = cs.load_credentials()
    assert creds == cs._DEFAULTS
    assert creds["alpaca_api_key"] == ""


def test_load_credentials_corrupt_file_falls_back_to_defaults(isolated_store):
    (isolated_store / "credentials.json").write_text("{not valid json")
    creds = cs.load_credentials()
    assert creds == cs._DEFAULTS


def test_save_and_load_roundtrip(isolated_store):
    saved = cs.save_credentials(
        api_key=f"  {FAKE_KEY}  ",  # whitespace should be stripped
        secret_key=FAKE_SECRET,
        base_url="https://paper-api.alpaca.markets/",  # trailing slash stripped
    )
    assert saved["alpaca_api_key"] == FAKE_KEY
    assert saved["alpaca_base_url"] == "https://paper-api.alpaca.markets"

    loaded = cs.load_credentials()
    assert loaded["alpaca_api_key"] == FAKE_KEY
    assert loaded["alpaca_secret_key"] == FAKE_SECRET


def test_save_credentials_partial_update_preserves_other_fields(isolated_store):
    cs.save_credentials(api_key=FAKE_KEY, secret_key=FAKE_SECRET)
    updated = cs.save_credentials(secret_key="new_secret_value")
    assert updated["alpaca_api_key"] == FAKE_KEY  # untouched
    assert updated["alpaca_secret_key"] == "new_secret_value"


def test_status_reports_masked_values_and_paper_flag(isolated_store):
    cs.save_credentials(api_key=FAKE_KEY, secret_key=FAKE_SECRET,
                         base_url="https://paper-api.alpaca.markets")
    result = cs.status()
    assert result["api_key_set"] is True
    assert result["secret_key_set"] is True
    assert result["is_paper"] is True
    assert FAKE_SECRET not in result["secret_key_masked"]
    assert result["api_key_masked"].startswith(FAKE_KEY[:4])


def test_status_live_url_is_not_flagged_as_paper(isolated_store):
    cs.save_credentials(api_key=FAKE_KEY, secret_key=FAKE_SECRET,
                         base_url="https://api.alpaca.markets")
    assert cs.status()["is_paper"] is False


def test_status_with_no_credentials_saved(isolated_store):
    result = cs.status()
    assert result["api_key_set"] is False
    assert result["secret_key_set"] is False
    assert result["api_key_masked"] == ""


def test_build_subprocess_env_injects_alpaca_vars_without_touching_os_environ(isolated_store):
    cs.save_credentials(api_key=FAKE_KEY, secret_key=FAKE_SECRET,
                         base_url="https://paper-api.alpaca.markets")
    base_env = {"PATH": "/usr/bin"}
    env = cs.build_subprocess_env(base_env=base_env)
    assert env["ALPACA_API_KEY"] == FAKE_KEY
    assert env["ALPACA_SECRET_KEY"] == FAKE_SECRET
    assert env["ALPACA_BASE_URL"] == "https://paper-api.alpaca.markets"
    # the caller's dict is not mutated in place
    assert "ALPACA_API_KEY" not in base_env
