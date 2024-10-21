import jax
import jax.numpy as jnp
from jax import lax

from typing import NamedTuple

# Type definitions for clarity
PRNGKey = jax.random.PRNGKey


class LSTMLayerParams(NamedTuple):
    w_xh_i: jax.Array
    w_xhb_i: jax.Array
    w_hh_i: jax.Array
    w_hhb_i: jax.Array
    w_xh_f: jax.Array
    w_xhb_f: jax.Array
    w_hh_f: jax.Array
    w_hhb_f: jax.Array
    w_xh_g: jax.Array
    w_xhb_g: jax.Array
    w_hh_g: jax.Array
    w_hhb_g: jax.Array
    w_xh_o: jax.Array
    w_xhb_o: jax.Array
    w_hh_o: jax.Array
    w_hhb_o: jax.Array


class LSTMParams(NamedTuple):
    layer_weights: list[LSTMLayerParams]


class EncoderConfig(NamedTuple):
    d_embed: int
    d_model: int
    d_src_vocab: int
    n_layers: int


class EncoderParams(NamedTuple):
    embeddings: jax.Array
    lstm_params: LSTMParams


class DecoderConfig(NamedTuple):
    d_embed: int
    d_model: int
    d_tgt_vocab: int
    n_layers: int


class DecoderParams(NamedTuple):
    embeddings: jax.Array
    lstm_params: LSTMParams
    classifier: jax.Array


class SeqToSeqConfig(NamedTuple):
    encoder_config: EncoderConfig
    decoder_config: DecoderConfig


class SeqToSeqParams(NamedTuple):
    encoder_params: EncoderParams
    # projects the d_model output of the encoder to an initial embedding input to pass
    # into the decoder
    output_embedding: jax.Array
    decoder_params: DecoderParams


class SeqToSeqVariables(NamedTuple):
    encoder_init_hidden_d: jax.Array
    encoder_init_cell_d: jax.Array


class SeqToSeq(NamedTuple):
    params: SeqToSeqParams
    variables: SeqToSeqVariables
    config: SeqToSeqConfig


# We use Xavier uniform initializer for all parameters of the model
param_initializer = jax.nn.initializers.glorot_uniform()


def init_lstm_layer(
    unused_rng_key: PRNGKey, d_input: int, d_output: int
) -> tuple[PRNGKey, LSTMLayerParams]:
    keys = jax.random.split(unused_rng_key, num=17)
    unused_key = keys[0]
    w_xh_i = param_initializer(keys[1], (d_input, d_output), jnp.float32)
    w_xhb_i = param_initializer(keys[2], (1, d_output), jnp.float32)
    w_hh_i = param_initializer(keys[3], (d_output, d_output), jnp.float32)
    w_hhb_i = param_initializer(keys[4], (1, d_output), jnp.float32)
    w_xh_f = param_initializer(keys[5], (d_input, d_output), jnp.float32)
    w_xhb_f = param_initializer(keys[6], (1, d_output), jnp.float32)
    w_hh_f = param_initializer(keys[7], (d_output, d_output), jnp.float32)
    w_hhb_f = param_initializer(keys[8], (1, d_output), jnp.float32)
    w_xh_g = param_initializer(keys[9], (d_input, d_output), jnp.float32)
    w_xhb_g = param_initializer(keys[10], (1, d_output), jnp.float32)
    w_hh_g = param_initializer(keys[11], (d_output, d_output), jnp.float32)
    w_hhb_g = param_initializer(keys[12], (1, d_output), jnp.float32)
    w_xh_o = param_initializer(keys[13], (d_input, d_output), jnp.float32)
    w_xhb_o = param_initializer(keys[14], (1, d_output), jnp.float32)
    w_hh_o = param_initializer(keys[15], (d_output, d_output), jnp.float32)
    w_hhb_o = param_initializer(keys[16], (1, d_output), jnp.float32)

    return unused_key, LSTMLayerParams(
        w_xh_i=w_xh_i,
        w_xhb_i=w_xhb_i,
        w_hh_i=w_hh_i,
        w_hhb_i=w_hhb_i,
        w_xh_f=w_xh_f,
        w_xhb_f=w_xhb_f,
        w_hh_f=w_hh_f,
        w_hhb_f=w_hhb_f,
        w_xh_g=w_xh_g,
        w_xhb_g=w_xhb_g,
        w_hh_g=w_hh_g,
        w_hhb_g=w_hhb_g,
        w_xh_o=w_xh_o,
        w_xhb_o=w_xhb_o,
        w_hh_o=w_hh_o,
        w_hhb_o=w_hhb_o,
    )


def init_encoder(
    unused_rng_key: PRNGKey, encoder_config: EncoderConfig
) -> tuple[PRNGKey, EncoderParams]:
    unused_key, key_embedding = jax.random.split(unused_rng_key, num=2)
    embeddings = param_initializer(
        key_embedding, (encoder_config.d_src_vocab, encoder_config.d_embed), jnp.float32
    )
    layer_weights = []
    for layer_idx in range(encoder_config.n_layers):
        d_input = encoder_config.d_embed if layer_idx == 0 else encoder_config.d_model
        d_output = encoder_config.d_model
        unused_key, cur_lstm_layer = init_lstm_layer(unused_key, d_input, d_output)
        layer_weights.append(cur_lstm_layer)

    lstm_params = LSTMParams(layer_weights=layer_weights)
    encoder_params = EncoderParams(embeddings=embeddings, lstm_params=lstm_params)
    return unused_key, encoder_params


def init_decoder(
    unused_rng_key: PRNGKey, decoder_config: DecoderConfig
) -> tuple[PRNGKey, DecoderParams]:
    unused_key, key_embedding, key_classifier = jax.random.split(unused_rng_key, num=3)
    embeddings = param_initializer(
        key_embedding, (decoder_config.d_tgt_vocab, decoder_config.d_embed), jnp.float32
    )
    layer_weights = []
    for layer_idx in range(decoder_config.n_layers):
        d_input = decoder_config.d_embed if layer_idx == 0 else decoder_config.d_model
        d_output = decoder_config.d_model
        unused_key, cur_lstm_layer = init_lstm_layer(unused_key, d_input, d_output)
        layer_weights.append(cur_lstm_layer)

    classifier_weights = param_initializer(
        key_classifier, (decoder_config.d_model, decoder_config.d_tgt_vocab)
    )

    lstm_params = LSTMParams(layer_weights=layer_weights)
    decoder_params = DecoderParams(
        embeddings=embeddings, lstm_params=lstm_params, classifier=classifier_weights
    )
    return unused_key, decoder_params


def init_seq_to_seq(
    unused_rng_key: PRNGKey, seq_to_seq_config: SeqToSeqConfig
) -> tuple[PRNGKey, SeqToSeq]:
    unused_key, encoder_params = init_encoder(
        unused_rng_key, seq_to_seq_config.encoder_config
    )
    unused_key, decoder_params = init_decoder(
        unused_key, seq_to_seq_config.decoder_config
    )
    unused_key, embedding_key = jax.random.split(unused_key, num=2)
    embedding_weights = param_initializer(
        embedding_key,
        (
            seq_to_seq_config.encoder_config.d_model,
            seq_to_seq_config.decoder_config.d_embed,
        ),
    )

    encoder_init_hidden_d = jnp.zeros((1, seq_to_seq_config.encoder_config.d_model))
    encoder_init_cell_d = jnp.zeros((1, seq_to_seq_config.encoder_config.d_model))
    seq_to_seq_variables = SeqToSeqVariables(
        encoder_init_hidden_d=encoder_init_hidden_d,
        encoder_init_cell_d=encoder_init_cell_d,
    )
    seq_to_seq_params = SeqToSeqParams(
        encoder_params=encoder_params,
        output_embedding=embedding_weights,
        decoder_params=decoder_params,
    )
    seq_to_seq = SeqToSeq(
        params=seq_to_seq_params,
        variables=seq_to_seq_variables,
        config=seq_to_seq_config,
    )

    return unused_key, seq_to_seq


def lstm_layer_apply(
    x_lbd: jax.Array,
    h_bd: jax.Array,
    c_bd: jax.Array,
    lstm_layer_params: LSTMLayerParams,
) -> tuple[jax.Array, jax.Array]:
    seq_len, batch_size, d_input = x_lbd.shape

    def _lstm_step(hc_bd, x_bd: jax.Array):
        h_bd, c_bd = hc_bd
        i_bd = jax.nn.sigmoid(
            jnp.matmul(x_bd, lstm_layer_params.w_xh_i)
            + lstm_layer_params.w_xhb_i
            + jnp.matmul(h_bd, lstm_layer_params.w_hh_i)
            + lstm_layer_params.w_hhb_i
        )
        f_bd = jax.nn.sigmoid(
            jnp.matmul(x_bd, lstm_layer_params.w_xh_f)
            + lstm_layer_params.w_xhb_f
            + jnp.matmul(h_bd, lstm_layer_params.w_hh_f)
            + lstm_layer_params.w_hhb_f
        )
        g_bd = jnp.tanh(
            jnp.matmul(x_bd, lstm_layer_params.w_xh_g)
            + lstm_layer_params.w_xhb_g
            + jnp.matmul(h_bd, lstm_layer_params.w_hh_g)
            + lstm_layer_params.w_hhb_g
        )
        o_bd = jax.nn.sigmoid(
            jnp.matmul(x_bd, lstm_layer_params.w_xh_o)
            + lstm_layer_params.w_xhb_o
            + jnp.matmul(h_bd, lstm_layer_params.w_hh_o)
            + lstm_layer_params.w_hhb_o
        )
        c_bd = jnp.multiply(f_bd, c_bd) + jnp.multiply(i_bd, g_bd)
        h_bd = jnp.multiply(o_bd, jnp.tanh(c_bd))

        return (h_bd, c_bd), (h_bd, c_bd)

    _, (h_lbd, c_lbd) = lax.scan(_lstm_step, (h_bd, c_bd), x_lbd)
    return h_lbd, c_lbd


def encoder_apply(
    x_bl: jax.Array,
    init_h_bd: jax.Array,
    init_c_bd: jax.Array,
    encoder_params: EncoderParams,
) -> tuple[jax.Array, jax.Array]:
    batch_size, seq_len = x_bl.shape
    # d_embed
    x_bld = encoder_params.embeddings[x_bl]
    x_lbd = jnp.transpose(x_bld, (1, 0, 2))
    input_lbd = x_lbd

    # d_model
    h_lbd = None
    c_lbd = None
    for lstm_layer_params in encoder_params.lstm_params.layer_weights:
        h_lbd, c_lbd = lstm_layer_apply(
            input_lbd, init_h_bd, init_c_bd, lstm_layer_params
        )
        input_lbd = h_lbd

    return h_lbd, c_lbd


def decoder_apply(
    x_bl: jax.Array,
    encoder_output_embedding_bd: jax.Array,
    init_h_bd: jax.Array,
    init_c_bd: jax.Array,
    decoder_params: DecoderParams,
) -> jax.Array:
    batch_size, seq_len = x_bl.shape
    # d_embed
    x_bld = decoder_params.embeddings[x_bl]
    x_lbd = jnp.transpose(x_bld, (1, 0, 2))
    # concat encoder output
    input_lbd = jnp.concatenate(
        [jnp.expand_dims(encoder_output_embedding_bd, 0), x_lbd]
    )

    # d_model
    h_lbd = None
    c_lbd = None
    for lstm_layer_params in decoder_params.lstm_params.layer_weights:
        h_lbd, c_lbd = lstm_layer_apply(
            input_lbd, init_h_bd, init_c_bd, lstm_layer_params
        )
        input_lbd = h_lbd

    # d_tgt_vocab
    logits_lbd = jnp.matmul(h_lbd, decoder_params.classifier)

    return logits_lbd


def compute_init_decoder_cell_state(
    c_lbd: jax.Array, src_mask_bl: jax.Array
) -> jax.Array:
    masked_hidden_bld = jnp.multiply(
        jnp.transpose(c_lbd, (1, 0, 2)), jnp.expand_dims(src_mask_bl, 2)
    )

    return jnp.mean(masked_hidden_bld, axis=1)


def seq_to_seq_apply(
    src_bl: jax.Array,
    src_mask_bl: jax.Array,
    tgt_bl: jax.Array,
    seq_to_seq_params: SeqToSeqParams,
    seq_to_seq_variables: SeqToSeqVariables,
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
    decoder_init_cell_state_bd = compute_init_decoder_cell_state(c_lbd, src_mask_bl)
    # d_embed of decoder
    encoder_output_embedding_bd = jnp.matmul(
        h_lbd[-1, ...], seq_to_seq_params.output_embedding
    )

    logits_lbd = decoder_apply(
        tgt_bl,
        encoder_output_embedding_bd,
        h_lbd[-1, ...],
        # use the same context vector computed from the encoder cell states at the
        # beginning of each layer
        decoder_init_cell_state_bd,
        seq_to_seq_params.decoder_params,
    )

    return jnp.transpose(logits_lbd, (1, 0, 2))
