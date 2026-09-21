import sys
import unittest

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from download_v4_a_confirm_archive_v1 import (
    open_checked_demo_stream,
    validate_download_hop_url,
)


SOURCE = "https://www.hltv.org/download/demo/2397691"


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
        self.requested.append(url)

        if kwargs.get("allow_redirects") is not False:
            raise AssertionError(
                "Automatic redirects must be disabled."
            )

        return self.responses.pop(0)


class RedirectSafetyTests(unittest.TestCase):

    def test_direct_hltv_response(self):
        response = FakeResponse(200)
        session = FakeSession([response])

        result, final_url = open_checked_demo_stream(
            session, SOURCE
        )

        self.assertIs(result, response)
        self.assertEqual(final_url, SOURCE)
        self.assertEqual(session.requested, [SOURCE])

    def test_internal_hltv_redirect(self):
        first = FakeResponse(
            302, "/download/demo/2397691?token=test"
        )
        second = FakeResponse(200)
        session = FakeSession([first, second])

        result, final_url = open_checked_demo_stream(
            session, SOURCE
        )

        self.assertIs(result, second)
        self.assertTrue(first.closed)
        self.assertEqual(
            final_url,
            SOURCE + "?token=test",
        )
        self.assertEqual(len(session.requested), 2)

    def test_external_cdn_is_not_contacted(self):
        first = FakeResponse(
            302,
            "https://cdn.example.org/archive.zip",
        )
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "SOURCE_REDIRECT_REVIEW_REQUIRED"
        ):
            open_checked_demo_stream(session, SOURCE)

        self.assertEqual(
            session.requested,
            [SOURCE],
        )
        self.assertTrue(first.closed)

    def test_http_redirect_is_not_contacted(self):
        first = FakeResponse(
            302,
            "http://www.hltv.org/archive.zip",
        )
        session = FakeSession([first])

        with self.assertRaises(RuntimeError):
            open_checked_demo_stream(session, SOURCE)

        self.assertEqual(session.requested, [SOURCE])

    def test_host_spoofing_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_download_hop_url(
                "https://www.hltv.org.evil.example/archive"
            )

    def test_redirect_loop_rejected(self):
        first = FakeResponse(302, SOURCE)
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "redirect loop"
        ):
            open_checked_demo_stream(session, SOURCE)

        self.assertEqual(session.requested, [SOURCE])

    def test_missing_location_rejected(self):
        first = FakeResponse(302)
        session = FakeSession([first])

        with self.assertRaisesRegex(
            RuntimeError, "no Location"
        ):
            open_checked_demo_stream(session, SOURCE)

        self.assertEqual(session.requested, [SOURCE])

    def test_initial_external_url_rejected_before_request(self):
        session = FakeSession([])

        with self.assertRaises(RuntimeError):
            open_checked_demo_stream(
                session,
                "https://cdn.example.org/archive.zip",
            )

        self.assertEqual(session.requested, [])

    def test_excessive_redirects_rejected(self):
        responses = [
            FakeResponse(
                302,
                f"/download/demo/2397691?hop={index}",
            )
            for index in range(6)
        ]

        session = FakeSession(responses)

        with self.assertRaisesRegex(
            RuntimeError, "too many redirects"
        ):
            open_checked_demo_stream(session, SOURCE)

        self.assertEqual(len(session.requested), 6)
        self.assertTrue(all(r.closed for r in responses))


if __name__ == "__main__":
    unittest.main()
