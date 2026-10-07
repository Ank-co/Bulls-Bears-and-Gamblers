"""Scoring checks on a tiny randomly initialized Qwen3 (no download, CPU only)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402

from bbg.llm_scoring import chat_prompt, encode, predict, score_batch  # noqa: E402
from bbg.prompts import ANSWERS, user_message  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
TWEETS = ["$AAPL beats estimates, shares jump", "Fed holds rates steady",
          "Oil plunges as demand outlook weakens on a much longer headline with many more words in it",
          "x"]


@pytest.fixture(scope="module")
def tiny():
    corpus = [user_message(t) for t in TWEETS] + list(ANSWERS.values()) * 20 + ["assistant user"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIALS,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>")
    tok.chat_template = TEMPLATE
    tok.padding_side = "left"
    cfg = transformers.Qwen3Config(vocab_size=tok.vocab_size + 8, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=512, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    return model, tok


def reference_scores(model, tok, text, answers=tuple(ANSWERS.values())):
    """Unbatched, unpadded, full-logits computation: the slow but obvious way."""
    out = []
    p_ids = tok(chat_prompt(tok, text), add_special_tokens=False)["input_ids"]
    for ans in answers:
        ids = tok(chat_prompt(tok, text) + ans, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
        out.append(sum(lp[i - 1, ids[i]].item() for i in range(len(p_ids), len(ids))))
    return torch.tensor(out)


def test_prompt_disables_thinking(tiny):
    _, tok = tiny
    assert chat_prompt(tok, "hello").endswith("<think>\n\n</think>\n\n")


def test_batched_left_padded_scores_match_reference(tiny):
    model, tok = tiny
    batched = score_batch(model, tok, TWEETS)
    for i, text in enumerate(TWEETS):
        assert torch.allclose(batched[i], reference_scores(model, tok, text), atol=1e-4), text


def test_answers_of_different_token_lengths(tiny):
    """Real tokenizers split the labels into different numbers of tokens: the mask must handle it."""
    model, tok = tiny
    answers = ["bullish", "zq", "neutral market outlook"]
    lengths = encode(tok, TWEETS[:1], answers).cand_lengths
    assert len(set(lengths)) == 3, lengths
    batched = score_batch(model, tok, TWEETS, answers)
    for i, text in enumerate(TWEETS):
        assert torch.allclose(batched[i], reference_scores(model, tok, text, answers), atol=1e-4)


def test_predict_is_independent_of_batch_size(tiny):
    model, tok = tiny
    p1, s1 = predict(model, tok, TWEETS, batch_size=1)
    p4, s4 = predict(model, tok, TWEETS, batch_size=4)
    assert p1 == p4 and torch.allclose(s1, s4, atol=1e-4)
    assert set(p1) <= set(ANSWERS)


def test_candidate_lengths_are_positive(tiny):
    _, tok = tiny
    enc = encode(tok, TWEETS[:1], list(ANSWERS.values()))
    assert all(n >= 1 for n in enc.cand_lengths)
