from .symbols import SYMBOLS_TO_ID


def cleaned_text_to_sequence(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    phones = [SYMBOLS_TO_ID[symbol] for symbol in cleaned_text]
    return phones
