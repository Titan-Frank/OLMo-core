from olmo_core.data import TokenizerConfig


def test_padded_vocab_size():
    assert TokenizerConfig.dolma2().padded_vocab_size() == 100352
    assert TokenizerConfig.gpt_neox_olmo_dolma_v1_5().padded_vocab_size() == 50304
    assert TokenizerConfig.qwen3_5().padded_vocab_size() == 248320


def test_from_hf():
    assert TokenizerConfig.from_hf("gpt2") == TokenizerConfig.gpt2()


def test_qwen3_5():
    assert TokenizerConfig.qwen3_5() == TokenizerConfig(
        vocab_size=248320,
        eos_token_id=248044,
        pad_token_id=248044,
        identifier="Qwen/Qwen3.5-9B",
    )
