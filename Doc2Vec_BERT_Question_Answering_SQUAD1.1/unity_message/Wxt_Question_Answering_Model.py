# -*- coding: utf-8 -*-
"""Wxt_Question_Answering_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KcWDmrizGcM0H8Gjc7gKEj9jOaoeMeZI


Question Answer Model - 
Uses the Doc2Vec for getting the question and the associated data set based on the user input
This question and data set are later passed BERT which is already pre-trained and fine tuned on SQUAD1.1 which returns the response

@Input - We need to train this model using Questions and find Reference Data set so that it can pass the same information to BERT

Author - Dhruv Shah
Last Updated - Septemeber 30th 2020
"""

#Install the Hugging Face Transformers Library for BERT Fine Tune Model
#!pip install transformers

import os
import numpy as np
import spacy
import gensim
import collections
import smart_open
import random
import sys
import torch
import textwrap
import glob

"""nlp = spacy.load('en_core_web_lg')

query = "Fetch my messages related to sports"

query = nlp(query)

entity = []

for token in query:
    if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
        entity.append(token.text.lower())

print(entity)
"""

# Set file names for train and test data
test_data_dir = '/home/ubuntu/Doc2Vec_BERT_Question_Answering_SQUAD1.1/Train_DataSet_Corpus'
lee_train_file = test_data_dir + os.sep + 'train_question.cor'
train_referencepara_file = test_data_dir + os.sep + 'train_referencepara.cor'
guide_reference_link = test_data_dir + os.sep + 'train_guide.cor'
#vocab_file = '/home/unity/IVR/ConversationalIvr/utility' + os.sep + 'text8'


def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def train_model():
    train_corpus = list(read_corpus(lee_train_file))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=1024, min_count=1, epochs=10000, window=2, dbow_words=1, dm=1)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec_unity_message")
    return model

def load_model():
    model_name = '/home/ubuntu/Doc2Vec_BERT_Question_Answering_SQUAD1.1/unity_message/doc2vec_unity_message'
    model = gensim.models.doc2vec.Doc2Vec.load(model_name)
    return model


def answer_question(question, answer_text):
    # Invoke the BERT Model and BERT Tokentizer with Question and Reference Text #
    
    ## Import the BERT for Question and Answer Model . This class fine tunes the model for benchmark data
    from transformers import BertForQuestionAnswering
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    
    ## Import the BERTTokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    #print('Answer: "' + answer + '"')
    return answer


def test_model(query):
    # Load the Corpus for Corpus for Model #
    #print("Reading Corpus for doc2vec")
    train_corpus = list(read_corpus(lee_train_file))
    train_referencepara_corpus = list(read_corpus(train_referencepara_file))
    train_guide_corpus = list (read_corpus(guide_reference_link))
    #print("Reading corpus completed")

    # Load the Pre-Trained Model on Question Answer Data Set #
    #print ('Load the Model')
    model = load_model()

    ranks = []
    second_ranks = []
    query = model.infer_vector(query.split(" "))

    # Comapre the Word Vectors for Trianed Model on same Training Corpus against the Query Vectors #
    sims = model.docvecs.most_similar([query], topn=len(model.docvecs))
    # Print the Similarity Score #
    #print (sims)

    # Enable the Question and Answers Set for DEBUG Purpose #
    #print('Question ({}): «{}»\n'.format(sims[0][0], ' '.join(train_corpus[sims[0][0]].words)))
    #print('Answer  ({}): «{}»\n'.format(sims[0][0], ' '.join(train_referencepara_corpus[sims[0][0]].words)))
    #print("Question Score",sims[0][1])
    referencelink = ""
    if sims[0][1] > 0.5:        
        bert_abstract = ' '.join(train_referencepara_corpus[sims[0][0]].words)
        question = ' '.join(train_corpus[sims[0][0]].words)
        for fileName in glob.glob(lee_train_file):
            with open(fileName) as f:
                for line in f.readlines():
                    if question.lower() in line.lower():
                        #print(question.lower())
                        #print(line.lower())
                        file = open(guide_reference_link)
                        all_lines = file.readlines()
                        referencelink = all_lines[sims[0][0]]
                        #print (referencelink)
                        #print(os.path.split(fileName))
                        #corVal = os.path.split(fileName)[-1].split('.')[0][-1]
                        #print(corVal)
                        #with open(test_data_dir + os.sep + 'train_guide' + '.cor') as completeText: 
                            #link_text = completeText.read()
                            #print(link_text)
                            #break
        # Load the Corpus for Matching Paragraph and Pass the Input squence to Model for correct Answer #
        answer = answer_question(question, bert_abstract)
        return answer + '\n' + 'Please refer below guide for detailed understanding : \n' + referencelink
    else:
        answer = "Sorry i am not very sure what to answer for this query"
        return answer


def main ():
    """ Train the model With Training Question and Answer Corpus
    Model Training is Required when we are are loading new content
    where word matrix have low understanding . for similar content
    no need to train the model again
    model = train_model()
    """

    #model = train_model()

    query = sys.argv[1]
    answer = test_model (query)
    print (answer)


if __name__ == "__main__":
        main()

