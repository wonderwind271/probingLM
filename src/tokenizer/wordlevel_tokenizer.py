"""A word-level tokenizer that only split on whitespace character."""

import json
import os
from collections import Counter
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class TrainableWordTokenizer(PreTrainedTokenizer):
    """Tokenizer class."""
    def __init__(
        self,
        vocab_file='',
        unk_token='[UNK]',
        pad_token='[PAD]',
        eos_token='[PAD]',
        **kwargs
    ):
        """Initialize with possible vocab file."""
        self._vocab = {unk_token: 0, pad_token: 1}
        self._ids_to_tokens = {0: unk_token, 1: pad_token}
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            **kwargs
        )
        if os.path.exists(vocab_file):
            with open(vocab_file) as fp:
                self._vocab = json.load(fp)
                self._ids_to_tokens = {value: key for key, value in self._vocab.items()}
        else:
            print('Warning: initializing from scratch. Expect `train_vocab` to be called')

    def train_vocab(
        self,
        texts: List[str] = [],
        text_file_path: Optional[str] = None,  # This must be a folder, not file
        max_vocab_size: Optional[int] = None,
        min_frequency: int = 1,
        save_file: Optional[str] = None
    ):
        """Train vocabulary from a list of texts."""
        # Tokenize all texts
        all_tokens = [token for text in texts for token in text.split()]
        if text_file_path:
            for root, _, files in os.walk(text_file_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                                content = file.read()
                                # Use regex to find all word sequences (letters, numbers, underscore)
                                words_in_file = content.split()
                                all_tokens.extend(words_in_file)
                        except Exception as e:
                            print(f'Error reading {file_path}: {e}')

        token_freqs = Counter(all_tokens)
        print('===Training Message===')
        print(token_freqs)
        # Filter tokens by minimum frequency
        filtered_tokens = [
            token for token, freq in token_freqs.items()
            if freq >= min_frequency
        ]
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(
            filtered_tokens,
            key=lambda x: token_freqs[x],
            reverse=True
        )

        # Limit vocabulary size if specified
        if max_vocab_size:
            sorted_tokens = sorted_tokens[:max_vocab_size]

        # Rebuild vocab dictionary
        base_idx = len(self._vocab)
        for token in sorted_tokens:
            if token not in self._vocab:
                self._vocab[token] = base_idx
                self._ids_to_tokens[base_idx] = token
                base_idx += 1
        if save_file:
            with open(save_file, 'w') as fp:
                json.dump(self._vocab, fp)
        print('===Training Message End===')

    def _tokenize(self, text: str) -> List[str]:
        """Split a given text into tokens using whitespace."""
        result = text.split()
        return result

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its corresponding ID."""
        return self._vocab.get(token, self._vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its corresponding token."""
        return self._ids_to_tokens.get(index, self.unk_token)
    
    def __len__(self) -> int:
        """Get vocab size with len."""
        return len(self.get_vocab())

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return all vocab."""
        return self._vocab


# Example usage
def test_trainable_tokenizer():
    """Test the TrainableWordTokenizer with example data."""
    tokenizer = TrainableWordTokenizer(vocab_file='vocab.json')

    # Training texts
    # texts = [
    #     'I love machine learning',
    #     'machine learning is fascinating',
    #     'deep learning transforms industries',
    #     'the:e child:e opens:e the:e box:e .:e there:d is:d nothing:d in:d the:d box:d .:d she:e closes:e it:e .:e'
    # ]
    tokenizer.train_vocab(text_file_path='data', save_file='vocab.json')
    # Train vocabulary
    # tokenizer.train_vocab(texts, min_frequency=1, save_file='vocab.json')

    # Demonstrate token-to-id conversion
    test_text = 'the:e child:e opens:e the:e box:e .:e there:d is:d nothing:d in:d the:d box:d .:d she:e closes:e it:e .:e I love machine learning'
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print('Tokens:', tokens)
    print('Token IDs:', token_ids)
    print('Vocabulary:', tokenizer._vocab)


# Optional: Uncomment to test
if __name__ == '__main__':
    test_trainable_tokenizer()
