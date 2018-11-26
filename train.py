from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import allennlp
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from dataset import BioMedReader
from allennlp.models.coreference_resolution.coref import CoreferenceResolver
torch.manual_seed(1)

#-----------------------------------dataset creation-------------------------------
max_span_width = 10
reader = BioMedReader(max_span_width)
train_dataset = reader.read("/home/surya/Documents/bio-coref/Bio-SCoRes/DATA/SPL/TRAIN")

validation_dataset = reader.read("/home/surya/Documents/bio-coref/Bio-SCoRes/DATA/SPL/TEST")
#-----------------------------------model creation---------------------------------

# vocab:Vocabulary
# text_field_embedder:TextFieldEmbedder
# Used to embed the text TextField we get as input to the model.

# context_layer:Seq2SeqEncoder
# This layer incorporates contextual information for each word in the document.

# mention_feedforward:FeedForward
# This feedforward network is applied to the span representations which is then scored by a linear layer.

# antecedent_feedforward: ``FeedForward``
# This feedforward network is applied to pairs of span representation, along with any pairwise features, which is then scored by a linear layer.

# feature_size: ``int``
# The embedding size for all the embedded features, such as distances or span widths.

# max_span_width: ``int``
# The maximum width of candidate spans.

# spans_per_word: float, required.
# A multiplier between zero and one which controls what percentage of candidate mention spans we retain with respect to the number of words in the document.

# max_antecedents: int, required.
# For each mention which survives the pruning stage, we consider this many antecedents.

# lexical_dropout: ``int``
# The probability of dropping out dimensions of the embedded text.

# initializer:InitializerApplicator, optional (default=``InitializerApplicator()``)
# Used to initialize the model parameters.

# regularizer:RegularizerApplicator, optional (default=``None``)
# If provided, will be used to calculate the regularization penalty during training.

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
EMBEDDING_DIM = 200
HIDDEN_DIM = 200
FEATURE_SIZE = 20
FEED_HIDDEN = 150
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
text_field_embedder = BasicTextFieldEmbedder({"tokens": token_embedding})
context_layer = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
mention_feedforward = allennlp.modules.feedforward.FeedForward(input_dim=HIDDEN_DIM,num_layers=2,hidden_dims=FEED_HIDDEN,activations=torch.nn.functional.relu)
antecedent_feedforward = allennlp.modules.feedforward.FeedForward(input_dim=HIDDEN_DIM,num_layers=2,hidden_dims=FEED_HIDDEN,activations=torch.nn.functional.relu)
feature_size = FEATURE_SIZE
max_span_width = max_span_width
spans_per_word = 0.4
max_antecedents = 250
lexical_dropout = 0.2

model = CoreferenceResolver(vocab,
                 text_field_embedder,
                 context_layer,
                 mention_feedforward,
                 antecedent_feedforward,
                 feature_size,
                 max_span_width,
                 spans_per_word,
                 max_antecedents,
                 lexical_dropout)

#-----------------------------------trainer creation--------------------------------
optimizer = optim.Adam(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)
trainer.train()
#-----------------------------------predictor---------------------------------------
# predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
# tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
# tag_ids = np.argmax(tag_logits, axis=-1)
# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
#------------------------------------------------------------------------------------