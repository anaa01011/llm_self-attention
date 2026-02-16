"""
HuggingFace-compatible tokenizer wrapper for SentencePiece.
Provides seamless integration with HuggingFace transformers library.
"""

from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
import sentencepiece as spm
from pathlib import Path


class ReasoningLLMTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for ReasoningLLM based on SentencePiece.
    Compatible with HuggingFace transformers ecosystem.
    """
    
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        **kwargs
    ):
        """
        Args:
            vocab_file: Path to the SentencePiece model file
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            unk_token: Unknown token
            pad_token: Padding token
        """
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(vocab_file)
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp_model.vocab_size()
    
    def get_vocab(self):
        """Return vocabulary as a dictionary."""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize a string into tokens."""
        return self.sp_model.encode_as_pieces(text)
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) to an id using the vocab."""
        return self.sp_model.piece_to_id(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an index (integer) to a token (str) using the vocab."""
        return self.sp_model.id_to_piece(index)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens (string) in a single string."""
        return self.sp_model.decode_pieces(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        """Save the vocabulary to a directory."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if filename_prefix:
            vocab_file = save_dir / f"{filename_prefix}-{self.vocab_files_names['vocab_file']}"
        else:
            vocab_file = save_dir / self.vocab_files_names['vocab_file']
        
        # Copy the model file
        import shutil
        shutil.copy(self.vocab_file, str(vocab_file))
        
        return (str(vocab_file),)
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by concatenating and adding special tokens.
        For single sequence: [BOS] X [EOS]
        For pair of sequences: [BOS] A [EOS] B [EOS]
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Get list where entries are [1] if a token is special, else [0].
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from sequences.
        Token type IDs are 0 for the first sequence, 1 for the second.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        
        if token_ids_1 is None:
            return len(bos + token_ids_0 + eos) * [0]
        return len(bos + token_ids_0 + eos) * [0] + len(token_ids_1 + eos) * [1]
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.sp_model.bos_id()
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.sp_model.eos_id()
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.sp_model.pad_id()
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.sp_model.unk_id()


def load_tokenizer(model_path: str) -> ReasoningLLMTokenizer:
    """
    Load a trained SentencePiece tokenizer.
    
    Args:
        model_path: Path to the .model file
        
    Returns:
        ReasoningLLMTokenizer instance
    """
    return ReasoningLLMTokenizer(vocab_file=model_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        tokenizer = load_tokenizer(sys.argv[1])
        
        test_text = "Let's think step by step to solve this problem."
        print(f"Text: {test_text}")
        
        # Encode
        encoded = tokenizer.encode(test_text)
        print(f"Encoded IDs: {encoded}")
        
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
        
        # Tokenize
        tokens = tokenizer.tokenize(test_text)
        print(f"Tokens: {tokens}")
    else:
        print("Usage: python tokenizer.py <path_to_tokenizer.model>")
