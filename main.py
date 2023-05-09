import tensorflow as tf
import tensorflow_text  # noqa
import tensorflow_hub as hub


# Loading models from tfhub.dev
encoder = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang/1")
preprocessor = hub.KerasLayer("https://tfhub.dev/jeongukjae/smaller_LaBSE_15lang_preprocess/1")

# Constructing model to encode texts into high-dimensional vectors
sentences = tf.keras.layers.Input(shape=(), dtype=tf.string, name="sentences")
encoder_inputs = preprocessor(sentences)
sentence_representation = encoder(encoder_inputs)["pooled_output"]
normalized_sentence_representation = tf.nn.l2_normalize(sentence_representation, axis=-1)  # for cosine similarity
model = tf.keras.Model(sentences, normalized_sentence_representation)
model.summary()

# Encoding multilingual sentences.
english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])
italian_sentences = tf.constant(["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."])
japanese_sentences = tf.constant(["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"])

english_embeds = model(english_sentences)
italian_embeds = model(italian_sentences)
japanese_embeds = model(japanese_sentences)

# English-Italian similarity
print(tf.tensordot(english_embeds, italian_embeds, axes=[[1], [1]]))

# English-Japanese similarity
print(tf.tensordot(english_embeds, japanese_embeds, axes=[[1], [1]]))

# Italian-Japanese similarity
print(tf.tensordot(italian_embeds, japanese_embeds, axes=[[1], [1]]))