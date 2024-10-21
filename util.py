import jax
import jax.numpy as jnp


def get_mask(
    tok_batch_bl: jax.Array, masked_token_idxs: list[int]
) -> tuple[jax.Array, jax.Array]:
    batch_size, _ = tok_batch_bl.shape
    combined_mask_bl = jnp.ones(tok_batch_bl.shape)
    for masked_token_idx in masked_token_idxs:
        cur_mask_bl = jnp.where(tok_batch_bl == masked_token_idx, 0, 1)
        combined_mask_bl = jnp.multiply(combined_mask_bl, cur_mask_bl)
    num_tokens_b = jnp.sum(combined_mask_bl, axis=-1)

    return combined_mask_bl, num_tokens_b


def convert_tokens_to_words(
    tokens_bl: jax.Array, vocab: dict[str, int]
) -> list[list[str]]:
    rev_vocab = {v: k for k, v in vocab.items()}
    batch_size, seq_len = tokens_bl.shape
    translations = []
    for sample_idx in range(batch_size):
        translation = []
        token_list = tokens_bl[sample_idx, :].tolist()
        for token_idx in token_list:
            translation.append(rev_vocab[token_idx])
        translations.append(translation)
    return translations


@jax.jit
def get_predicted_logits(masked_logits_bld: jax.Array, tgt_bl: jax.Array) -> jax.Array:
    batch_size, seq_len = tgt_bl.shape
    _, d_tgt_vocab = masked_logits_bld.shape
    masked_logits_bld = jnp.reshape(
        masked_logits_bld, (batch_size, seq_len, d_tgt_vocab)
    )
    softmaxed_masked_logits_bld = jax.nn.softmax(masked_logits_bld)
    return jnp.argmax(softmaxed_masked_logits_bld, axis=-1)


def get_training_time_predictions(
    masked_logits_bld: jax.Array, tgt_bl: jax.Array, vocab
) -> list[list[str]]:
    prediction_token_idx = get_predicted_logits(masked_logits_bld, tgt_bl)
    return convert_tokens_to_words(prediction_token_idx, vocab)
