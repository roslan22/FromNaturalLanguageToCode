import os
from utils import Utils

DATA_PATH = "conala-corpus/"
CONALA_MINED = "conala-mined.jsonl"
TRAIN_FILE = "conala-train.json"
TEST_FILE = "conala-test.json"
EMBEDDING_NAME = "none.json"

class DataProvider(object):
    """Data provider for conala dataset"""

    def __init__(self, shuffle_order=False, rng=None):
        """ Create provider object.
        """
        self.shuffle_order = shuffle_order
        self.rng = rng
        self.data_file_names = { 
            'conala_mined': CONALA_MINED,
            'train' : TRAIN_FILE,
            'test'  : TEST_FILE,
        }
        self.embeddings_names = {
            'embed_by_asin' : EMBEDDING_NAME
        }

    def load_data(self, which_set):
        assert which_set in ['conala_mined', 'train', 'test'], (
            'Expected which_set to be either conala_mined, train, or test. '
            'Got {0}'.format(which_set)
        )

        file_name = self.data_file_names[which_set]
        return Utils.load_json(DATA_PATH, file_name)    

    """
	def load_embeddings(self, which_embedding):
        assert which_embedding in ['embed_by_asin'], (
            'Expected which_embedding to be embed_by_asin. '
            'Got {0}'.format(which_embedding)
        )

        file_name = self.embeddings_names[which_embedding]
        return Utils.load_obj(DATA_PATH, file_name)
	"""