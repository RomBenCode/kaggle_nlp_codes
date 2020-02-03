from nltk.corpus import stopwords


EN_STOP_WORDS = stopwords.words('english')

# Embedding-only model constants
EMBEDDING_DIM = 300
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOK = "<OOV>"
# length of input sentences in nb of characters.
SENT_INPUT_LENGTH = 70

