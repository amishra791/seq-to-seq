import jax
import jax.numpy as jnp

from data import get_dataset, get_training_dataloader, PAD_TOKEN
from functools import partial

from lstm_model import (
    SeqToSeq,
    DecoderParams,
    SeqToSeqParams,
    SeqToSeqConfig,
    SeqToSeqVariables,
    seq_to_seq_apply,
    EncoderConfig,
    DecoderConfig,
    init_seq_to_seq,
    encoder_apply,
    compute_context_vector,
)


def decoder_greedy_decode(
    # d_embed
    init_input_bd: jax.Array,
    # d_model
    init_h_lbd: jax.Array,
    init_c_lbd: jax.Array,
    decoder_params: DecoderParams,
    max_len: int = 100,
) -> jax.Array:

    next_token_predictions_lst = []
    input_bd = init_input_bd
    h_lbd = init_h_lbd  # (num_layers, batch_size, d_model)
    c_lbd = init_c_lbd
    for i in range(max_len):
        h_bd_lst, c_bd_lst = [], []
        for layer_idx, lstm_layer_params in enumerate(
            decoder_params.lstm_params.layer_weights
        ):
            h_bd, c_bd = h_lbd[layer_idx, ...], c_lbd[layer_idx, ...]
            i_bd = jax.nn.sigmoid(
                jnp.matmul(input_bd, lstm_layer_params.w_xh_i)
                + lstm_layer_params.w_xhb_i
                + jnp.matmul(h_bd, lstm_layer_params.w_hh_i)
                + lstm_layer_params.w_hhb_i
            )
            f_bd = jax.nn.sigmoid(
                jnp.matmul(input_bd, lstm_layer_params.w_xh_f)
                + lstm_layer_params.w_xhb_f
                + jnp.matmul(h_bd, lstm_layer_params.w_hh_f)
                + lstm_layer_params.w_hhb_f
            )
            g_bd = jnp.tanh(
                jnp.matmul(input_bd, lstm_layer_params.w_xh_g)
                + lstm_layer_params.w_xhb_g
                + jnp.matmul(h_bd, lstm_layer_params.w_hh_g)
                + lstm_layer_params.w_hhb_g
            )
            o_bd = jax.nn.sigmoid(
                jnp.matmul(input_bd, lstm_layer_params.w_xh_o)
                + lstm_layer_params.w_xhb_o
                + jnp.matmul(h_bd, lstm_layer_params.w_hh_o)
                + lstm_layer_params.w_hhb_o
            )
            c_bd = jnp.multiply(f_bd, c_bd) + jnp.multiply(i_bd, g_bd)
            h_bd = jnp.multiply(o_bd, jnp.tanh(c_bd))

            h_bd_lst.append(h_bd)
            c_bd_lst.append(c_bd)
            input_bd = h_bd

        h_lbd = jnp.stack(h_bd_lst)
        c_lbd = jnp.stack(c_bd_lst)

        # d_tgt_vocab
        logits_bd = jnp.matmul(h_lbd[-1, ...], decoder_params.classifier)
        softmaxed_logits_bd = jax.nn.softmax(logits_bd)
        # d_model
        next_token_predictions_bl = jnp.argmax(
            softmaxed_logits_bd, axis=-1, keepdims=True
        )
        next_token_predictions_lst.append(next_token_predictions_bl)
        # d_embed for input at next timestep
        input_bd = jnp.squeeze(
            decoder_params.embeddings[next_token_predictions_bl], axis=1
        )

    return jnp.concatenate(next_token_predictions_lst, axis=1)


@partial(jax.jit, static_argnums=3)
def seq_to_seq_greedy_decode(
    src_bl: jax.Array,
    seq_to_seq_params: SeqToSeqParams,
    seq_to_seq_variables: SeqToSeqVariables,
    src_vocab_pad_tok_idx: int,
) -> jax.Array:

    batch_size, seq_len = src_bl.shape
    encoder_init_h_bd = jnp.repeat(
        seq_to_seq_variables.encoder_init_hidden_d, batch_size, axis=0
    )
    encoder_init_c_bd = jnp.repeat(
        seq_to_seq_variables.encoder_init_cell_d, batch_size, axis=0
    )
    h_lbd, c_lbd = encoder_apply(
        src_bl, encoder_init_h_bd, encoder_init_c_bd, seq_to_seq_params.encoder_params
    )

    # d_model
    src_mask_bl = jnp.where(src_bl != src_vocab_pad_tok_idx, 1, 0)
    context_bd = compute_context_vector(c_lbd, src_mask_bl)
    # d_embed of decoder
    encoder_output_embedding_bd = jnp.matmul(
        h_lbd[-1, ...], seq_to_seq_params.output_embedding
    )

    num_layers = len(seq_to_seq_params.decoder_params.lstm_params.layer_weights)
    next_token_predictions_bl = decoder_greedy_decode(
        encoder_output_embedding_bd,
        jnp.repeat(jnp.expand_dims(h_lbd[-1, ...], axis=0), num_layers, axis=0),
        jnp.repeat(jnp.expand_dims(context_bd, axis=0), num_layers, axis=0),
        seq_to_seq_params.decoder_params,
    )

    return next_token_predictions_bl


if __name__ == "__main__":
    dataset, max_len = get_dataset(num_vocab=10000)
    src_vocab, tgt_vocab = dataset.src_vocab, dataset.tgt_vocab
    dataloader = get_training_dataloader(dataset, max_len=max_len, batch_size=64)

    key = jax.random.key(0)
    encoder_config = EncoderConfig(
        d_embed=128, d_model=256, d_src_vocab=len(dataset.src_vocab), n_layers=2
    )
    decoder_config = DecoderConfig(
        d_embed=128, d_model=256, d_tgt_vocab=len(dataset.tgt_vocab), n_layers=2
    )
    seq_to_seq_config = SeqToSeqConfig(
        encoder_config=encoder_config, decoder_config=decoder_config
    )

    _, seq_to_seq = init_seq_to_seq(key, seq_to_seq_config)

    for src, tgt in dataloader:
        print(src.shape, tgt.shape)
        src_mask_bl = jnp.where(src != src_vocab[PAD_TOKEN], 1, 0)
        logits_bld = seq_to_seq_apply(
            src, src_mask_bl, tgt, seq_to_seq.params, seq_to_seq.variables
        )
        print(logits_bld.shape)
        next_token_predictions_bl = seq_to_seq_greedy_decode(
            src, seq_to_seq.params, seq_to_seq.variables, src_vocab[PAD_TOKEN]
        )
        print(next_token_predictions_bl.shape)
