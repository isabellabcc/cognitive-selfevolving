from pipeline.sft import pad_features, tokenize_prompt_completion


class FakeTokenizer:
    eos_token_id = 99
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10, 11, 12] if text == "question" else [20, 21]


def test_prompt_and_padding_are_never_supervised():
    feature = tokenize_prompt_completion(FakeTokenizer(), "question", "answer", 20)
    assert feature["input_ids"] == [10, 11, 12, 20, 21, 99]
    assert feature["labels"] == [-100, -100, -100, 20, 21, 99]
    batch = pad_features([feature, {
        "input_ids": [10, 20], "attention_mask": [1, 1], "labels": [-100, 20],
    }], 0)
    assert batch["labels"][1] == [-100, 20, -100, -100, -100, -100]
    assert batch["attention_mask"][1] == [1, 1, 0, 0, 0, 0]
