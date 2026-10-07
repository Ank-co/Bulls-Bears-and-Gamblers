"""Shared fixtures: a tiny randomly initialized Qwen3 saved on disk (no download, CPU only)."""
import pytest

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_qwen_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from bbg.prompts import ANSWERS, user_message

    corpus = ([user_message(f"$AAPL shares jump rally plunge drop fed holds update {i}") for i in range(5)]
              + list(ANSWERS.values()) * 20)
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    cfg = transformers.Qwen3Config(vocab_size=tok.vocab_size + 8, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=512, tie_word_embeddings=False,
                                   # default 0.02 makes the output head too weak for a 64-dim random
                                   # model to express a peaked distribution at all
                                   initializer_range=0.3)
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-qwen3")
    transformers.Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path
