#Instaladores

# Packages used.
import os # Used for setting Keras configuration variables.
os.environ["KERAS_BACKEND"] = "jax" # Set a parameter for Keras.
import re # Used for splitting text on whitespace.

import keras # Used for defining an training the model.
import pandas as pd # Used for loading the dataset.
import tensorflow as tf # Used for shuffling the dataset.

# Used for displaying nicer error messages.
from IPython.display import display, HTML
from ai_foundations import training # For training your model.
from ai_foundations import generation # For prompting your model.
from ai_foundations import visualizations # For visualizing probabilities.
from ai_foundations.feedback.course_1 import slm # For providing feedback.

# The following line provides configuration for Keras.
keras.utils.set_random_seed(812)  # For Keras layers.

#Descarga del dataset
africa_galore = pd.read_json(
    "https://storage.googleapis.com/dm-educational/assets/ai_foundations/africa_galore.json"
)
dataset = africa_galore["description"].values
print("Loaded dataset with", dataset.shape[0], "paragraphs.")

#Capa de tokenizacion
class SimpleWordTokenizer:
    """A simple word tokenizer.

    The tokenizer splits the text sequence based on whitespace, using the
    `encode` method to convert the text into a sequence of indices and the
    `decode` method to convert indices back into text.

    The simple word tokenizer that can be initialized with a corpus or using a
    provided vocabulary list

    Typical usage example:

        corpus = "Hello there!"
        tokenizer = SimpleWordTokenizer(text)
        print(tokenizer.encode('Hello'))

    """

    # Define constants.
    UNKNOWN_TOKEN = "<UNK>"
    PAD_TOKEN = "<PAD>"

    def __init__(self, corpus: list[str], vocabulary: list[str] | None = None):
        """Initializes the tokenizer with texts in corpus or with a vocabulary.

        Args:
          corpus: Input text dataset.
          vocabulary: A pre-defined vocabulary. If None,
              the vocabulary is automatically inferred from the texts.
        """

        if vocabulary is None:
            # Build the vocabulary from scratch.
            if isinstance(corpus, str):
                corpus = [corpus]

            # Convert text sequence to tokens.
            tokens = []
            for text in corpus:
                for token in self.space_tokenize(text):
                    tokens.append(token)

            # Create a vocabulary comprising of unique tokens.
            vocabulary = self.build_vocabulary(tokens)

            # Add special unknown and pad tokens to the vocabulary list.
            self.vocabulary = (
                [self.PAD_TOKEN] + vocabulary + [self.UNKNOWN_TOKEN]
            )

        else:
            self.vocabulary = vocabulary

        # Size of vocabulary.
        self.vocabulary_size = len(self.vocabulary)

        # Create token-to-index and index-to-token mappings.
        self.token_to_index = {}
        self.index_to_token = {}
        # Loop through all tokens in the vocabulary. enumerate automatically
        # assigns a unique index to each token.
        for index, token in enumerate(self.vocabulary):
            self.token_to_index[token] = index
            self.index_to_token[index] = token

        # Map the special tokens to their IDs.
        self.pad_token_id = self.token_to_index[self.PAD_TOKEN]
        self.unknown_token_id = self.token_to_index[self.UNKNOWN_TOKEN]

    def space_tokenize(self, text: str) -> list[str]:
        """Splits a given text on whitespace into tokens.

        Args:
            text: Text to split on whitespace.

        Returns:
            List of tokens after splitting `text`.
        """

        # Use re.split such that multiple spaces are treated as a single
        # separator.
        return re.split(" +", text)

    def join_text(self, text_list: list[str]) -> str:
        """Combines a list of tokens into a single string.

        The combined tokens, as a single string, are separated by spaces in the
        string.

        Args:
            text_list: List of tokens to be joined.

        Returns:
            String with all tokens joined with a whitespace.

        """
        return " ".join(text_list)

    def build_vocabulary(self, tokens: list[str]) -> list[str]:
        """Create a vocabulary list from the list of tokens.

        Args:
            tokens: The list of tokens in the dataset.

        Returns:
            List of unique tokens (vocabulary) in the dataset.
        """
        return sorted(list(set(tokens)))

    def encode(self, text: str) -> list[int]:
        """Encodes a text sequence into a list of indices.

        Args:
            text: The input text to be encoded.

        Returns:
            A list of indices corresponding to the tokens in the input text.
        """

        # Convert tokens into indices.
        indices = []
        unk_index = self.token_to_index[self.UNKNOWN_TOKEN]
        for token in self.space_tokenize(text):
            token_index = self.token_to_index.get(token, unk_index)
            indices.append(token_index)

        return indices

    def decode(self, indices: int | list[int]) -> str:
        """Decodes a list (or single index) of integers back into tokens.

        Args:
            indices: A single index or a list of indices to be
                decoded into tokens.

        Returns:
            A string of decoded tokens corresponding to the input indices.
        """

        # If a single integer is passed, convert it into a list.
        if isinstance(indices, int):
            indices = [indices]

        # Map indices to tokens.
        tokens = []
        for index in indices:
            token = self.index_to_token.get(index, self.unknown_token_id)
            tokens.append(token)

        # Join the decoded tokens into a single string.
        return self.join_text(tokens)


# Initialize the tokenizer. This will build the tokenizer's vocabulary with
# all the tokens that appear in the dataset.
tokenizer = SimpleWordTokenizer(dataset)

# Translate all tokens to their corresponding IDs.
encoded_tokens = []
for text in dataset:
    # Split text into tokens and translate the tokens to token IDs.
    token_ids = tokenizer.encode(text)
    encoded_tokens.append(token_ids)

encoded_tokens[0][:10]

print(f"Length of first paragraph: {len(encoded_tokens[0]):,}")

# Compute the length of the shortest paragraph.
shortest_paragraph_length = len(min(encoded_tokens, key=len))

# Compute the length of the longest paragraph.
longest_paragraph_length = len(max(encoded_tokens, key=len))

print("Length of the shortest paragraph is:", shortest_paragraph_length)
print("Length of the longest paragraph is:", longest_paragraph_length)

# @title Set `max_length` for padding and truncating data.

max_length = 200  # @param {type: "number"}

if max_length <= 0:
    display(
        HTML(
            f"<h3>Error:</h3><p>Max length must be greater than 0. Please"
            f" increase <code>max_length</code>.</p><p></p>"
        )
    )

elif max_length > longest_paragraph_length:
    display(
        HTML(
            f"<h3>Error:</h3><p>The padding token <code>"
            f" {tokenizer.pad_token_id}</code> will be added to all"
            f" sequences - you probably don't want that. Please reduce"
            f" <code>max_length</code>.</p><p></p>"
        )
    )

else:
    if max_length < longest_paragraph_length:
        display(
            HTML(
                f"<p><strong>Note:</strong> The longest paragraph has"
                f" {longest_paragraph_length} tokens,"
                f" but <code>max_length</code> is set to {max_length}."
                f" Paragraphs longer than <code>max_length</code> will be"
                " truncated.</p><p></p>"
            )
        )

    padded_sequences = keras.preprocessing.sequence.pad_sequences(
        encoded_tokens,
        maxlen=max_length,
        padding="post",
        truncating="post",
        value=tokenizer.pad_token_id,
    )

    print("New length of first paragraph:", len(padded_sequences[0]), "\n")

    print(
        "Padding makes the length of all sequences the same as the specified"
        " `max_length`."
    )

    print(
        "Notice the padded token IDs {tokenizer.pad_token_id} appearing at the"
        f" end of the sequence.\n"
    )
    print("Padded tokens of first paragraph:\n", padded_sequences[0])

# Prepare input and target for the transformer model.
# For each example, extract all tokens except the last one.
input_sequences = padded_sequences[:, :-1]
# For each example, extract all tokens except the first one.
target_sequences = padded_sequences[:, 1:]

print("First 10 token IDs in first input sequence:", input_sequences[0, :10])
print(
    "First 10 tokens in first input sequence:",
    tokenizer.decode(input_sequences[0, :10]),
)

print("\n")

print("First 10 token IDs in first target sequence:", target_sequences[0, :10])
print(
    "First 10 tokens in target sequence:",
    tokenizer.decode(target_sequences[0, :10])
)
max_length = input_sequences.shape[1]

# Create TensorFlow dataset to prepare sequences.
tf_dataset = tf.data.Dataset.from_tensor_slices((input_sequences, target_sequences))

# Randomly shuffle the dataset.
# The buffer_size determines how many examples from the dataset
# are held in memory before shuffling.
# If you are working with a very large dataset,
# reduce the buffer_size as needed.
tf_dataset = tf_dataset.shuffle(buffer_size=len(input_sequences))

# Specify batch size.
batch_size = 32  # @param {type: "number"}

# Create batches.
batches = tf_dataset.batch(batch_size)

for batch in batches.take(1):
    print(batch)

total_batches = 0
for batch in batches:
    total_batches += 1
print("Total number of batches is:", total_batches)

model = training.create_model(
    max_length=max_length,
    vocabulary_size=tokenizer.vocabulary_size,
    learning_rate=1e-4
)

prompt = "Abeni,"
prompt_ids = tokenizer.encode(prompt)
text_gen_callback = training.TextGenerator(
    max_tokens=10, start_tokens=prompt_ids, tokenizer=tokenizer, print_every=10
)

#num_epochs = 200  # @param {type: "number"}
# verbose=2: Instructs the model.fit method to print one line per
# epoch so you see how the loss is decreasing and generated texts improving.
#history = model.fit(
#    x=batches, verbose=2, epochs=num_epochs, callbacks=[text_gen_callback]
#)
prompt = "also is a african" #@param {type: "string"}
num_tokens_to_generate = 20 #@param {type: "number"}
generated_text, probs = generation.generate_text(
    prompt,
    num_tokens_to_generate,
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.pad_token_id,
    sampling_mode="greedy" # To generate the highest probability generation.
)


print("Generated text:", generated_text)
print("\n")

visualizations.plot_next_token(probs[0], prompt=prompt, tokenizer=tokenizer)