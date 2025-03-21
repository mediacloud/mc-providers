"""
tests for OnlineNewsMediaCloudProvider._check_response
which interprets elasticsearch_dsl Response objects

[Phil: If you know me, I almost NEVER write static tests (IMO the
cost/benefit is high most of the time), but errors don't grow on
trees, and the code is complex and brittle, so here we are!]

These tests are DEFINITELY too strict/rigid but they're a starting place!

Should be runnable via:
venv/bin/pip install python-dotenv pytest # only needed once
venv/bin/pytest mc_providers/test/test_onlinenews_errors.py
"""

import logging

from elasticsearch_dsl.response import Response

from mc_providers.exceptions import MysteryProviderException, PermanentProviderException, ProviderException, TemporaryProviderException
from mc_providers.onlinenews import OnlineNewsMediaCloudProvider

def check(p, d):
    r = Response(None, d)
    p._check_response(r)

provider = OnlineNewsMediaCloudProvider()

def test_parse_error(caplog):
    try:
        check(provider, {
            "took": 65,
            "timed_out": False,
            "_shards": {
                "total": 150,
                "successful": 17,
                "skipped": 17,
                "failed": 133,
                "failures": [
                    {
                        "reason": {
                            "type": "query_shard_exception",
                            "reason": "Failed to parse query [(]",
                            "caused_by": {
                                "type": "parse_exception",
                                "reason": "parse_exception: Cannot parse '(': Encountered \"<EOF>\" at line 1, column 1.\nWas expecting one of:\n"
                            }
                        }
                    }
                ]
            }
        })
    except PermanentProviderException as e:
        assert str(e).startswith("parse_exception:")
        assert len(caplog.records) == 0
        return
    assert False

def test_permanent_breaker(caplog):
    caplog.set_level(logging.INFO)
    try:
        check(provider, {
            "took": 65,
            "timed_out": False,
            "_shards": {
                "total": 150,
                "successful": 17,
                "skipped": 17,
                "failed": 133,
                "failures": [
                    {
                        "shard": 10,
                        "reason": {
                            "type": "circuit_breaking_exception",
                            "reason": "[fielddata] Data too large, data for [Global Ordinals] would be [12890341545/12gb], which is larger than the limit of [12884901888/12gb]",
                            "durability": "PERMANENT"
                        }
                    }
                ]
            }
        })
    except PermanentProviderException as e:
        assert str(e) == "circuit_breaking_exception"
        assert len(caplog.records) == 2
        assert caplog.records[0].levelno == logging.INFO
        assert caplog.records[0].message == "MC._search 133/150 shards failed; reasons: Counter({'circuit_breaking_exception': 1})"
        assert caplog.records[1].message.startswith("permanent error")
        return
    assert False

def test_temporary_breaker(caplog):
    caplog.set_level(logging.INFO)
    try:
        check(provider, {
            "took": 65,
            "timed_out": False,
            "_shards": {
                "total": 150,
                "successful": 17,
                "skipped": 17,
                "failed": 133,
                "failures": [
                    {
                        "reason": {
                            "type": "circuit_breaking_exception",
                            "reason": "[fielddata] Data too large....",
                        }
                    }
                ]
            }
        })
    except TemporaryProviderException as e:
        assert str(e) == "Out of memory"
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO
        assert caplog.records[0].message == "MC._search 133/150 shards failed; reasons: Counter({'circuit_breaking_exception': 1})"
        return

    assert False

def test_mystery(caplog):
    caplog.set_level(logging.INFO)
    try:
        check(provider, {
            "took": 65,
            "timed_out": False,
            "_shards": {
                "total": 150,
                "successful": 17,
                "skipped": 0,
                "failed": 133,
                "failures": [
                    {
                        "reason": {
                            "type": "some_other_exception",
                            "reason": "blah blah blah"
                        }
                    }
                ]
            }
        })
    except MysteryProviderException as e:
        assert str(e) == "some_other_exception"
        assert len(caplog.records) == 2
        assert caplog.records[0].levelno == logging.INFO
        assert caplog.records[0].message == "MC._search 133/150 shards failed; reasons: Counter({'some_other_exception': 1})"
        assert caplog.records[1].levelno == logging.ERROR
        assert caplog.records[1].message.startswith("Unknown response error")
        return
    assert False

def test_timed_out(caplog):
    try:
        check(provider, {
            # NOTE! *ONLY* the data Response.success() looks at!!!
            "timed_out": True,
            "_shards": {
                "total": 0,
                "successful": 0,
            }
        })
    except TemporaryProviderException as e:
        assert str(e) == "Timed out"
        assert len(caplog.records) == 0
        return
    assert False
