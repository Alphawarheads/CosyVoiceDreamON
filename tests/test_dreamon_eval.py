"""Evaluation accounting tests; standard library only."""
from pathlib import Path
import tempfile
import unittest
import wave

from cosyvoice.utils.dreamon_eval import read_eval_meta, write_eval_meta, prompt_rate_length, scoring_pairs, summarize_scores
from tools.prepare_dreamon_eval import prepare_rows


class EvaluationTests(unittest.TestCase):
    def test_local_pairs_use_distinct_same_speaker_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = ("1_a", "1_b", "2_a", "2_b")
            for utt in ids:
                with wave.open(str(root / f"{utt}.wav"), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8000)
                    handle.writeframes(b"\x00\x00" * 24000)
            (root / "wav.scp").write_text("".join(f"{utt} {root / (utt + '.wav')}\n" for utt in ids), encoding="utf-8")
            (root / "text").write_text("".join(f"{utt} sentence {utt}\n" for utt in ids), encoding="utf-8")
            (root / "utt2spk").write_text("".join(f"{utt} {utt[0]}\n" for utt in ids), encoding="utf-8")
            rows = prepare_rows(root, 2)
            self.assertEqual([row.utt for row in rows], ["1_b", "2_b"])
            self.assertTrue(all(row.prompt_wav != row.target_wav for row in rows))
            self.assertTrue(all(row.prompt_wav.stem[0] == row.utt[0] for row in rows))

    def test_metadata_round_trip_and_invalid_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.wav").write_bytes(b"audio")
            meta = root / "meta.lst"
            meta.write_text("sample.wav|hello there|prompt.wav|good morning\n", encoding="utf-8")
            rows = read_eval_meta(meta)
            self.assertEqual(rows[0].utt, "sample")
            write_eval_meta(root / "copy.lst", rows)
            self.assertEqual(read_eval_meta(root / "copy.lst"), rows)
            for content in ("../escape|hello|prompt.wav|target\n",
                            "x|hello|prompt.wav|target|prompt.wav\n",
                            "x|hello|prompt.wav|target\nx.wav|hello|prompt.wav|other\n"):
                meta.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_eval_meta(meta)

    def test_duration_estimate(self):
        self.assertEqual(prompt_rate_length("one two three four", "one two", 50, 750), (100, False))
        self.assertEqual(prompt_rate_length("你好世界", "你好", 50, 75), (75, True))

    def test_missing_audio_and_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompt.wav").write_bytes(b"audio")
            meta = root / "meta.lst"
            meta.write_text("a|prompt|prompt.wav|target\n", encoding="utf-8")
            rows = read_eval_meta(meta)
            with self.assertRaises(FileNotFoundError):
                scoring_pairs(rows, root)
            (root / "a.wav").write_bytes(b"audio")
            self.assertEqual(len(scoring_pairs(rows, root)), 1)
            raw = root / "scores.txt"
            raw.write_text("a.wav\t0\ttarget\ttarget\t0\t0\t0\n", encoding="utf-8")
            self.assertEqual(summarize_scores(raw, "wer", 1), 0)
            with self.assertRaises(ValueError):
                summarize_scores(raw, "wer", 2)
            raw.write_text("a.wav\t2\ttarget\twrong\t1\t0\t1\n", encoding="utf-8")
            self.assertEqual(summarize_scores(raw, "wer", 1), 2)
            raw.write_text("a|b\t0.8\navg score: 0.8", encoding="utf-8")
            self.assertEqual(summarize_scores(raw, "sim", 1), 0.8)


if __name__ == "__main__":
    unittest.main()
