"""Offline data-path tests: no model inference or third-party packages required."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as experiment


class DataTests(unittest.TestCase):
    def test_parser(self):
        rows = experiment.parse_ebmsass(b'id1\tSentence A\tSentence B\t3.4\n', 'outcome')
        self.assertEqual(rows[0]['score'], 3.4)
        self.assertEqual(rows[0]['category'], 'outcome')
        self.assertEqual(rows[0]['pair_id'], 'id1')

    def test_bad_rows(self):
        for payload in (b'id\ta\tb\t6', b'id\ta\tb\tnan', b'id\ta\tb', b'id\t\tb\t2'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                experiment.parse_ebmsass(payload, 'background')

    def test_download_cache_selection_and_checksum(self):
        def fake_fetch(url):
            if 'api.github.com' in url:
                return json.dumps({'sha': 'test-commit'}).encode()
            return ''.join(f'{i}\tSentence A\tSentence B\t2\n' for i in range(200)).encode()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(experiment, 'ROOT', root), patch.object(experiment, 'fetch', side_effect=fake_fetch) as fetch:
                manifest = experiment.prepare_data(['ebmsass'])
                self.assertEqual(fetch.call_count, 6)
                self.assertEqual(manifest['ebmsass']['expected_rows'], 1000)
                self.assertEqual(len(manifest['ebmsass']['files']), 5)
                fetch.reset_mock()
                self.assertEqual(experiment.prepare_data(['ebmsass']), manifest)
                fetch.assert_not_called()
                first = root / 'data' / manifest['ebmsass']['files'][0]['local']
                first.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    experiment.prepare_data(['ebmsass'])


if __name__ == '__main__':
    unittest.main()
