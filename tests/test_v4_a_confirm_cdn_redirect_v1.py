"""Offline tests: fake HTTP responses; no network requests."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_v4_a_confirm_archive_v1 import (
    open_checked_demo_stream,
)

ORIGIN = "https://www.hltv.org/download/demo/111689"
CDN = "https://r2-demos.hltv.org/example-archive"


class FakeResponse:
    def __init__(self, status, location=None):
        self.status_code = status
        self.headers = (
            {"Location": location} if location is not None else {}
        )
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        return self.responses[len(self.requested) - 1]


class RedirectTests(unittest.TestCase):
    def test_single_approved_cdn_redirect(self):
        first = FakeResponse(302, CDN)
        second = FakeResponse(200)
        session = FakeSession([first, second])

        response, final_url = open_checked_demo_stream(
            session, ORIGIN
        )

        self.assertIs(response, second)
        self.assertEqual(final_url, CDN)
        self.assertEqual(session.requested, [ORIGIN, CDN])
        self.assertTrue(first.closed)

    def test_unrelated_external_host_is_not_requested(self):
        session = FakeSession([
            FakeResponse(302, "https://other.example/archive")
        ])

        with self.assertRaises(RuntimeError):
            open_checked_demo_stream(session, ORIGIN)

        self.assertEqual(session.requested, [ORIGIN])

    def test_cdn_cannot_redirect_again(self):
        session = FakeSession([
            FakeResponse(302, CDN),
            FakeResponse(302, "https://other.example/archive"),
        ])

        with self.assertRaisesRegex(
            RuntimeError, "CDN attempted another redirect"
        ):
            open_checked_demo_stream(session, ORIGIN)

        self.assertEqual(session.requested, [ORIGIN, CDN])

    def test_cdn_query_and_port_are_rejected(self):
        for destination in (
            CDN + "?token=example",
            "https://r2-demos.hltv.org:444/example-archive",
            "http://r2-demos.hltv.org/example-archive",
        ):
            with self.subTest(destination=destination):
                session = FakeSession([
                    FakeResponse(302, destination)
                ])

                with self.assertRaises(RuntimeError):
                    open_checked_demo_stream(session, ORIGIN)

                self.assertEqual(session.requested, [ORIGIN])

    def test_direct_cdn_entry_is_rejected(self):
        session = FakeSession([])

        with self.assertRaises(RuntimeError):
            open_checked_demo_stream(session, CDN)

        self.assertEqual(session.requested, [])

    def test_original_hltv_endpoint_without_redirect(self):
        response = FakeResponse(200)
        session = FakeSession([response])

        returned, final_url = open_checked_demo_stream(
            session, ORIGIN
        )

        self.assertIs(returned, response)
        self.assertEqual(final_url, ORIGIN)
        self.assertEqual(session.requested, [ORIGIN])


if __name__ == "__main__":
    unittest.main()
