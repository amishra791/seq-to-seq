import jax
from jax import lax
import jax.numpy as jnp

from util import get_mask

# vmapped methods to make the batch dimension work properly
vmap_isin = jax.vmap(jnp.isin)
vmap_take = jax.vmap(jnp.take)


@jax.jit
def calculate_top1_acc(
    masked_logits_bld: jax.Array, tgt_bl: jax.Array, masked_token_idxs: tuple[int]
):
    batch_size, seq_len = tgt_bl.shape
    _, d_tgt_vocab = masked_logits_bld.shape
    masked_logits_bld = jnp.reshape(
        masked_logits_bld, (batch_size, seq_len, d_tgt_vocab)
    )
    softmaxed_masked_logits_bld = jax.nn.softmax(masked_logits_bld)
    _, top_idx_blk = jax.lax.top_k(softmaxed_masked_logits_bld, 1)
    is_in_topk_bl = vmap_isin(
        tgt_bl.flatten(), lax.collapse(top_idx_blk, 0, 2)
    ).reshape((batch_size, seq_len))
    combined_mask_bl, num_tokens_b = get_mask(tgt_bl, masked_token_idxs)

    return jnp.multiply(is_in_topk_bl, combined_mask_bl).sum() / num_tokens_b.sum()


@jax.jit
def calculate_top3_acc(
    masked_logits_bld: jax.Array, tgt_bl: jax.Array, masked_token_idxs: tuple[int]
):
    batch_size, seq_len = tgt_bl.shape
    _, d_tgt_vocab = masked_logits_bld.shape
    masked_logits_bld = jnp.reshape(
        masked_logits_bld, (batch_size, seq_len, d_tgt_vocab)
    )
    softmaxed_masked_logits_bld = jax.nn.softmax(masked_logits_bld)
    _, top_idx_blk = jax.lax.top_k(softmaxed_masked_logits_bld, 3)
    is_in_topk_bl = vmap_isin(
        tgt_bl.flatten(), lax.collapse(top_idx_blk, 0, 2)
    ).reshape((batch_size, seq_len))
    combined_mask_bl, num_tokens_b = get_mask(tgt_bl, masked_token_idxs)

    return jnp.multiply(is_in_topk_bl, combined_mask_bl).sum() / num_tokens_b.sum()


@jax.jit
def calculate_top5_acc(
    masked_logits_bld: jax.Array,
    tgt_bl: jax.Array,
    masked_token_idxs: tuple[int],
):
    batch_size, seq_len = tgt_bl.shape
    _, d_tgt_vocab = masked_logits_bld.shape
    masked_logits_bld = jnp.reshape(
        masked_logits_bld, (batch_size, seq_len, d_tgt_vocab)
    )
    softmaxed_masked_logits_bld = jax.nn.softmax(masked_logits_bld)
    _, top_idx_blk = jax.lax.top_k(softmaxed_masked_logits_bld, 5)
    is_in_topk_bl = vmap_isin(
        tgt_bl.flatten(), lax.collapse(top_idx_blk, 0, 2)
    ).reshape((batch_size, seq_len))
    combined_mask_bl, num_tokens_b = get_mask(tgt_bl, masked_token_idxs)

    return jnp.multiply(is_in_topk_bl, combined_mask_bl).sum() / num_tokens_b.sum()


@jax.jit
def calculate_perplexity(
    masked_logits_bld: jax.Array, tgt_bl: jax.Array, masked_token_idxs: tuple[int]
):
    batch_size, seq_len = tgt_bl.shape
    _, d_tgt_vocab = masked_logits_bld.shape
    masked_logits_bld = jnp.reshape(
        masked_logits_bld, (batch_size, seq_len, d_tgt_vocab)
    )
    log_softmax_probs_bld = jax.nn.log_softmax(masked_logits_bld)
    softmax_probs_bld = jax.nn.softmax(masked_logits_bld)
    log_probs_bl = jnp.reshape(
        jnp.take_along_axis(
            lax.collapse(log_softmax_probs_bld, 0, 2),
            jnp.expand_dims(tgt_bl.flatten(), axis=1),
            axis=1,
        ),
        (batch_size, seq_len),
    )
    combined_mask_bl, num_tokens_b = get_mask(tgt_bl, masked_token_idxs)
    masked_log_probs_bl = jnp.multiply(log_probs_bl, combined_mask_bl)

    batch_perplexities_b = jnp.exp(
        -jnp.sum(masked_log_probs_bl, axis=-1) / num_tokens_b
    )
    return jnp.mean(batch_perplexities_b)
