"""Data sources: text windows and packed prompt/response batches from a growing JSONL file."""
import json
import threading
import time

import pytest
import torch

import bitlattice as bl

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.train_from_iterator(["question answer the cat sat on the mat"] * 20,
                           trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "</s>"],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="</s>")


def test_text_windows_are_deterministic():
    ids = torch.arange(1000)
    a = next(iter(bl.data.TextWindows(ids, 16, 3, seed=1)))[0]
    b = next(iter(bl.data.TextWindows(ids, 16, 3, seed=1)))[0]
    assert a.shape == (3, 17) and torch.equal(a, b)


def test_jsonl_stream_masks_prompts_and_follows_a_growing_file(tok, tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("")

    def producer():
        with open(path, "a") as f:
            for i in range(40):
                f.write(json.dumps({"prompt": f"question {i}", "response": "the cat sat on the mat"}) + "\n")
                f.flush()
                time.sleep(0.005)

    th = threading.Thread(target=producer)
    th.start()
    it = iter(bl.data.JsonlStream(str(path), tok, seq=32, batch=2, poll_s=0.01))
    x, y = next(it)
    th.join()
    assert x.shape == (2, 33) and y.shape == (2, 33)
    target = tok("the cat sat on the mat</s>", add_special_tokens=False).input_ids
    row = y[0].tolist()
    trained = [t for t in row if t != bl.data.IGNORE]
    assert trained[:len(target)] == target                     # only responses carry labels
    assert (y[0] == bl.data.IGNORE).any()
    ex_x, ex_y = bl.data.encode_example(tok, {"prompt": "question 0", "response": "the cat sat on the mat"})
    assert x[0, :len(ex_x)].tolist() == ex_x
