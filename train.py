import jax
from jax import lax
import jax.numpy as jnp
import optax
import random

from torch.utils.data import DataLoader
from data import get_dataset, get_training_dataloader, PAD_TOKEN
from lstm_model import (
    EncoderConfig,
    DecoderConfig,
    SeqToSeqConfig,
    init_seq_to_seq,
    SeqToSeqParams,
    SeqToSeqVariables,
    SeqToSeq,
    seq_to_seq_apply,
)
from decode import seq_to_seq_greedy_decode
from metrics import (
    calculate_top1_acc,
    calculate_top3_acc,
    calculate_top5_acc,
    calculate_perplexity,
)
from util import convert_tokens_to_words, get_training_time_predictions
from functools import partial

jax.config.update('jax_log_compiles', True)


def loss_fn(
    seq_to_seq_params: SeqToSeqParams,
    seq_to_seq_variables: SeqToSeqVariables,
    src_bl: jax.Array,
    src_vocab_pad_tok_idx: int,
    tgt_bl: jax.Array,
    tgt_vocab_pad_tok_idx: int,
) -> tuple[jax.Array, jax.Array]:
    batch_size, seq_len = tgt_bl.shape
    src_mask_bl = jnp.where(src_bl != src_vocab_pad_tok_idx, 1, 0)
    logits_bld = seq_to_seq_apply(
        src_bl, src_mask_bl, tgt_bl, seq_to_seq_params, seq_to_seq_variables
    )[:, :-1, :]
    cross_ent_bl = optax.softmax_cross_entropy_with_integer_labels(
        lax.collapse(logits_bld, 0, 2), tgt_bl.flatten()
    )

    # zero out predictions corresponding to <PAD> tokens as input
    tgt_pad_mask_bl = jnp.where(tgt_bl == tgt_vocab_pad_tok_idx, 0, 1)
    num_tokens = jnp.sum(tgt_pad_mask_bl)
    masked_cross_ent_bl = jnp.multiply(cross_ent_bl, lax.collapse(tgt_pad_mask_bl, 0))
    masked_logits_bld = jnp.multiply(
        lax.collapse(logits_bld, 0, 2),
        jnp.expand_dims(lax.collapse(tgt_pad_mask_bl, 0), 1),
    )
    avg_loss = jnp.sum(masked_cross_ent_bl / num_tokens)
    return avg_loss, masked_logits_bld


def train(
    seq_to_seq: SeqToSeq,
    dataloader: DataLoader,
    src_vocab: dict[str, int],
    tgt_vocab: dict[str, int],
    num_epochs: int = 45,
) -> None:
    scheduler = optax.exponential_decay(
        init_value=4e-3, transition_steps=1000, decay_rate=0.99
    )

    # Combining gradient transforms using `optax.chain`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
        optax.scale_by_adam(),  # Use the updates from adam.
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(seq_to_seq.params)

    @partial(jax.jit, static_argnums=[3, 5])
    def train_step(
        seq_to_seq_params: SeqToSeqParams,
        seq_to_seq_variables: SeqToSeqVariables,
        src_bl: jax.Array,
        src_vocab_pad_tok_idx: int,
        tgt_bl: jax.Array,
        tgt_vocab_pad_tok_idx: int,
        opt_state: optax.OptState,
    ) -> tuple[SeqToSeqParams, optax.OptState, jax.Array, jax.Array]:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, masked_logits_bld), grads = grad_fn(
            seq_to_seq_params,
            seq_to_seq_variables,
            src_bl,
            src_vocab_pad_tok_idx,
            tgt_bl,
            tgt_vocab_pad_tok_idx,
        )

        updates, opt_state = optimizer.update(grads, opt_state)
        updated_seq_to_seq_params = optax.apply_updates(seq_to_seq_params, updates)

        return updated_seq_to_seq_params, opt_state, loss, masked_logits_bld

    for epoch_num in range(num_epochs):
        for idx, batch_bl in enumerate(dataloader):
            print(f"Epoch: {epoch_num}, Idx: {idx}")
            src_bl, tgt_bl = batch_bl
            updated_params, opt_state, loss, masked_logits_bld = train_step(
                seq_to_seq.params,
                seq_to_seq.variables,
                src_bl,
                src_vocab[PAD_TOKEN],
                tgt_bl,
                tgt_vocab[PAD_TOKEN],
                opt_state,
            )
            next_token_greedy_predictions_bl = seq_to_seq_greedy_decode(
                src_bl, seq_to_seq.params, seq_to_seq.variables, src_vocab[PAD_TOKEN]
            )

            seq_to_seq = SeqToSeq(
                params=updated_params,
                variables=seq_to_seq.variables,
                config=seq_to_seq.config,
            )

            print(f"Loss: {loss}")
            print(
                f"Top 1 acc: {calculate_top1_acc(
                    masked_logits_bld,
                    tgt_bl,
                    (tgt_vocab[PAD_TOKEN],),
                )}"
            )
            print(
                f"Top 3 acc: {calculate_top3_acc(
                    masked_logits_bld,
                    tgt_bl,
                    (tgt_vocab[PAD_TOKEN],),
                )}"
            )
            print(
                f"Top 5 acc: {calculate_top5_acc(
                    masked_logits_bld,
                    tgt_bl,
                    (tgt_vocab[PAD_TOKEN],),
                )}"
            )
            print(
                f"Perplexity: {calculate_perplexity(
                    masked_logits_bld,
                    tgt_bl,
                    (tgt_vocab[PAD_TOKEN],),
                )}"
            )
            src_translations = convert_tokens_to_words(src_bl, src_vocab)
            pred_tgt_translations = get_training_time_predictions(
                masked_logits_bld, tgt_bl, tgt_vocab
            )
            tgt_translations = convert_tokens_to_words(tgt_bl, tgt_vocab)
            greedy_decode_translations = convert_tokens_to_words(
                next_token_greedy_predictions_bl, tgt_vocab
            )
            print("Printing Translations:")
            rand_idx = random.randrange(0, len(tgt_translations))
            print(f"Original: {" ".join(src_translations[rand_idx][:30])}")
            print(f"Actual Translation: {" ".join(tgt_translations[rand_idx][:30])}")
            print(f"Teacher Forced Translation: {" ".join(pred_tgt_translations[rand_idx][:30])}")
            print(f"Greedy decoded Translation: {" ".join(greedy_decode_translations[rand_idx][:30])}")


if __name__ == "__main__":

    dataset, max_len = get_dataset(num_vocab=10000)
    src_vocab, tgt_vocab = dataset.src_vocab, dataset.tgt_vocab
    print(len(dataset.src_vocab), len(dataset.tgt_vocab))
    dataloader = get_training_dataloader(dataset, max_len=max_len, batch_size=128)

    key = jax.random.key(0)
    encoder_config = EncoderConfig(
        d_embed=256, d_model=512, d_src_vocab=len(dataset.src_vocab), n_layers=2
    )
    decoder_config = DecoderConfig(
        d_embed=256, d_model=512, d_tgt_vocab=len(dataset.tgt_vocab), n_layers=2
    )
    seq_to_seq_config = SeqToSeqConfig(
        encoder_config=encoder_config, decoder_config=decoder_config
    )
    _, seq_to_seq = init_seq_to_seq(key, seq_to_seq_config)

    train(seq_to_seq, dataloader, src_vocab, tgt_vocab)
