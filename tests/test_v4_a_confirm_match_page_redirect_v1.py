
import sys
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_v4_a_confirm_demo_source_v1 import (
    open_checked_match_page,
    validate_match_page_hop_url,
)


MATCH_ID = "2397691"

ORIGINAL = (
    "https://www.hltv.org/matches/2397691/"
    "nodwin-clutch-series-12-swiss-round-4-1-"
    "nodwin-clutch-series-12"
)

RENAMED = (
    "https://www.hltv.org/matches/2397691/"
    "brute-vs-sparta-nodwin-clutch-series-12"
)


class FakeResponse:

    def __init__(self, status_code, location=None):
        self.status_code = status_code
        self.headers = {}
        self.closed = False

        if location is not None:
            self.headers["Location"] = location

    def close(self):
        self.closed = True


class FakeSession:

    def __init__(self, responses):
        self.responses = list(responses)
        self.requested = []

    def get(self, url, **kwargs):
        assert kwargs.get("allow_redirects") is False
        self.requested.append(url)
        return self.responses.pop(0)


class MatchPageRedirectTests(unittest.TestCase):

    def test_direct_response(self):
        result = FakeResponse(200)
        session = FakeSession([result])

        response, final_url = open_checked_match_page(
            session, ORIGINAL, MATCH_ID
        )

        self.assertIs(response, result)
        self.assertEqual(final_url, ORIGINAL)
        self.assertEqual(session.requested, [ORIGINAL])

    def test_same_match_id_new_slug(self):
        first = FakeResponse(301, RENAMED)
        second = FakeResponse(200)
        session = FakeSession([first, second])

        response, final_url = open_checked_match_page(
            session, ORIGINAL, MATCH_ID
        )

        self.assertIs(response, second)
        self.assertEqual(final_url, RENAMED)
        self.assertEqual(
            session.requested, [ORIGINAL, RENAMED]
        )
        self.assertTrue(first.closed)

    def test_external_host_not_contacted(self):
        first = FakeResponse(
            302, "https://outside.example/matches/2397691/x"
        )
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "SOURCE_REVIEW_REQUIRED"
        ):
            open_checked_match_page(
                session, ORIGINAL, MATCH_ID
            )

        self.assertEqual(session.requested, [ORIGINAL])
        self.assertTrue(first.closed)

    def test_other_match_id_not_contacted(self):
        first = FakeResponse(
            302,
            "https://www.hltv.org/matches/9999999/other",
        )
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "different Match ID"
        ):
            open_checked_match_page(
                session, ORIGINAL, MATCH_ID
            )

        self.assertEqual(session.requested, [ORIGINAL])

    def test_http_destination_not_contacted(self):
        first = FakeResponse(
            302,
            "http://www.hltv.org/matches/2397691/other",
        )
        session = FakeSession([first])

        with self.assertRaises(RuntimeError):
            open_checked_match_page(
                session, ORIGINAL, MATCH_ID
            )

        self.assertEqual(session.requested, [ORIGINAL])

    def test_invalid_initial_url_not_contacted(self):
        session = FakeSession([])

        with self.assertRaises(RuntimeError):
            open_checked_match_page(
                session,
                "https://www.hltv.org.evil.example/"
                "matches/2397691/other",
                MATCH_ID,
            )

        self.assertEqual(session.requested, [])

    def test_redirect_loop_stops(self):
        first = FakeResponse(302, ORIGINAL)
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "redirect loop"
        ):
            open_checked_match_page(
                session, ORIGINAL, MATCH_ID
            )

        self.assertEqual(session.requested, [ORIGINAL])
        self.assertTrue(first.closed)


if __name__ == "__main__":
    unittest.main()
