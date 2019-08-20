import os
import time
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_datasets as tfds
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
from IPython.display import clear_output
from nltk.tokenize import WordPunctTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

output_dir = "/data/acp18dj/"
topic_vocab = os.path.join(output_dir, "topic_vocab")
abstract_vocab = os.path.join(output_dir, "abstract_vocab")
checkpoint_path = os.path.join(output_dir, "checkpoints")
log_path = os.path.join(output_dir, 'logs')

topic = []
abstract = []
dirPath1 = "/data/acp18dj/msc_dataset/lr/"
dirPath2 = "/data/acp18dj/msc_dataset/topic/"
dirPath3 = "/data/acp18dj/ug_dataset/lr/"
dirPath4 = "/data/acp18dj/ug_dataset/topic/"

lr_files = [f for f in os.listdir(dirPath1) if os.path.isfile(os.path.join(dirPath1, f))]
lr_files = sorted(lr_files)
for fname in lr_files:
    if ("txt" not in fname):
        continue
    with open(dirPath1+fname, "r", encoding='utf-8') as fp:
        abstract.append(fp.read())

tp_files = [f for f in os.listdir(dirPath2) if os.path.isfile(os.path.join(dirPath2, f))]
tp_files = sorted(tp_files)
for fname in tp_files:
    if ("txt" not in fname):
        continue
    if (fname not in lr_files):
        continue
    with open(dirPath2+fname, "r", encoding='utf-8') as fp:
        topic.append(fp.read())

print('msc topic size:')
print(len(topic))
print('msc lr size:')
print(len(abstract))

lr_files = [f for f in os.listdir(dirPath3) if os.path.isfile(os.path.join(dirPath3, f))]
lr_files = sorted(lr_files)
for fname in lr_files:
    if ("txt" not in fname):
        continue
    with open(dirPath3+fname, "r", encoding='utf-8') as fp:
        abstract.append(fp.read())

tp_files = [f for f in os.listdir(dirPath4) if os.path.isfile(os.path.join(dirPath4, f))]
tp_files = sorted(tp_files)
for fname in tp_files:
    if ("txt" not in fname):
        continue
    if (fname not in lr_files):
        continue
    with open(dirPath4+fname, "r", encoding='utf-8') as fp:
        topic.append(fp.read())

print('ug topic size:')
print(len(topic))
print('ug lr size:')
print(len(abstract))

dirPath = "/data/acp18dj/msc_dataset/topic/"
files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
files = sorted(files)
for fname in files:
    if ("txt" not in fname):
        continue
    with open(dirPath+fname, "r", encoding='utf-8') as fp:
        topic.append(fp.read())

dirPath = "/data/acp18dj/ug_dataset/topic/"
files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
files = sorted(files)
for fname in files:
    if ("txt" not in fname):
        continue
    with open(dirPath+fname, "r", encoding='utf-8') as fp:
        topic.append(fp.read())
print(len(topic))

dirPath = "/data/acp18dj/msc_dataset/abstract/"
files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
files = sorted(files)
for fname in files:
    if ("txt" not in fname):
        continue
    with open(dirPath+fname, "r", encoding='utf-8') as fp:
        abstract.append(fp.read())

dirPath = "/data/acp18dj/ug_dataset/abstract/"
files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
files = sorted(files)
for fname in files:
    if ("txt" not in fname):
        continue
    with open(dirPath+fname, "r", encoding='utf-8') as fp:
        abstract.append(fp.read())
print(len(abstract))

data_size = len(topic)
train_size = int(0.8 * data_size)
test_size = data_size - train_size
print('training size:', train_size)
print('test size', test_size)

# Transfer data list to tensor
examples = tf.data.Dataset.from_tensor_slices((topic, abstract))

# Split training data and testing data
train_examples = examples.take(train_size)
test_examples = examples.skip(train_size)

for top, abt in test_examples.take(3):
    print(top)
    print(abt)
    print('-' * 10)

tokenizer_topic = tfds.features.text.Tokenizer()
topic_vocabulary_set = set()
for top, abt in examples:
    some_tokens = tokenizer_topic.tokenize(top.numpy())
    topic_vocabulary_set.update(some_tokens)
tokenizer_topic.save_to_file(topic_vocab)
topic_vocab_size = len(topic_vocabulary_set)
print('topic vocab_size:', topic_vocab_size)

tokenizer_abstract = tfds.features.text.Tokenizer()
abstract_vocabulary_set = set()
for top, abt in examples:
    some_tokens = tokenizer_abstract.tokenize(abt.numpy())
    abstract_vocabulary_set.update(some_tokens)
tokenizer_abstract.save_to_file(abstract_vocab)
abstract_vocab_size = len(abstract_vocabulary_set)
print('abstract vocab_size:', abstract_vocab_size)

topic_encoder = tfds.features.text.TokenTextEncoder(topic_vocabulary_set)
abstract_encoder = tfds.features.text.TokenTextEncoder(abstract_vocabulary_set)

def encode(to_t, ab_t):
    # tokenizer_summ.vocab_size for the <start> token
    # tokenizer_summ.vocab_size + 1 for the <end> token
    to_indices = [topic_vocab_size] + topic_encoder.encode(
                                    to_t.numpy()) + [topic_vocab_size + 1]
    ab_indices = [abstract_vocab_size] + abstract_encoder.encode(
                                    ab_t.numpy()) + [abstract_vocab_size + 1]
    return to_indices, ab_indices

def tf_encode(to_t, ab_t):
    # force 'su_t' and 'ne_t' to Eager Tensors by py_function
    return tf.py_function(encode, [to_t, ab_t], [tf.int64, tf.int64])

MAX_LENGTH = 500
BATCH_SIZE = 4
BUFFER_SIZE = 5000

train_dataset = (train_examples  # output：(Summary, News)
                 .map(tf_encode) # output：(Summary index sequence, News index sequence)
#                  .filter(filter_max_length)
                 .cache() # speed up
                 .shuffle(BUFFER_SIZE) # random dataset
                 .padded_batch(BATCH_SIZE, # padding to same length for each batch
                               padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE)) # speed up

test_dataset = (test_examples
               .map(tf_encode)
#                .filter(filter_max_length)
               .padded_batch(BATCH_SIZE,
                             padded_shapes=([-1], [-1])))


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    # tf.cast change data type
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# mask the future tokens in a sequence
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    # scale matmul_qk
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # get seq_k sequence length
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # scale by sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # weighted average
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# initially assign the output dimension 'd_model' & 'num_heads'
# output.shape            == (batch_size, seq_len_q, d_model)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # how many heads divided by d_model
        self.d_model = d_model # base dimension before split_heads

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # new dimension for each head

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)  # linear transformation after concatenating heads

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # divide 'd_model' into 'num_heads' depth
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # concatenate 'num_heads' depth to original dimension 'd_model'
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  # two linear transformations for input, add ReLU activation func
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # one for sub-layer, one for layer norm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # one for sub-layer, one for layer norm
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # sub-layer 1: MHA
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # sub-layer 2: FFN
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        # Masked multi-head attention (with look ahead mask and padding mask)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # Multi-head attention (with padding mask).
        # V (value) and K (key) receive the encoder output as inputs.
        # Q (query) receives the output from the masked multi-head attention sublayer.
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, combined_mask, inp_padding_mask):
        # all sub-layers output: (batch_size, target_seq_len, d_model)
        # enc_output is Encoder output sequence: (batch_size, input_seq_len, d_model)
        # attn_weights_block_1: (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weights_block_2: (batch_size, num_heads, target_seq_len, input_seq_len)

        # sub-layer 1: Decoder layer
        attn1, attn_weights_block1 = self.mha1(x, x, x, combined_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # sub-layer 2: Decoder layer focuses the Encoder final output
        # (batch_size, target_seq_len, d_model) (V, K, Q)
        # attention weights: the importance given to the decoder's input based on the encoder's output
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, inp_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # sub-layer 3: FFN
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    # - num_layers: how many EncoderLayers
    # - input_vocab_size: transfer index to vector
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # (input_dim, output_dim)
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # x.shape == (batch_size, input_seq_len)
        # all layer output: (batch_size, input_seq_len, d_model)
        input_seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]

        # combine embedding and positional encoder, and regularization
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,  rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
        # FFN output logits number, represent the probability passing softmax
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        # Decoder output pass the last linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
input_vocab_size = topic_vocab_size + 2
target_vocab_size = abstract_vocab_size + 2
dropout_rate = 0.15
print("input_vocab_size:", input_vocab_size)
print("target_vocab_size:", target_vocab_size)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # comput cross entropy of all position, but not sum up
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  # only compute the loss of non <pad> position

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # the default warmup_steps = 4000
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate)

print(f"""Transformer has {num_layers} Encoder / Decoder layers
d_model: {d_model}
num_heads: {num_heads}
dff: {dff}
input_vocab_size: {input_vocab_size}
target_vocab_size: {target_vocab_size}
dropout_rate: {dropout_rate}
""")

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_{train_size}data"

checkpoint_dir = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_path, run_id)

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

    last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print(f'load the up-to-date checkpoint, the model has already trained {last_epoch} epochs.')
else:
    last_epoch = 0
    print("No checkpoint, start training.")

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    # use Adam optimizer for updating parameters of Transformer
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # the loss and training acc recorded on TensorBoard
    train_loss(loss)
    train_accuracy(tar_real, predictions)

epoch = last_epoch
while(1):
    start = time.time()

    # reset TensorBoard metrics
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        #print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          #        epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    # save checkpoint for each epoch
    if (epoch + 1) % 1 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    epoch += 1
    if (train_loss.result() < 1):
        break
