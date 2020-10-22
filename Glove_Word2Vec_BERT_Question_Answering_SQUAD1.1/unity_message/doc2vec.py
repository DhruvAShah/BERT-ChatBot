import gensim
import os
import collections
import smart_open
import random

# Set file names for train and test data
#test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
test_data_dir = '/home/unity/IVR/ConversationalIvr/utility/gensim'
lee_train_file = test_data_dir + os.sep + 'train_question.cor'
lee_test_file = test_data_dir + os.sep + 'test_question.cor'
train_answer_file = test_data_dir + os.sep + 'train_answer.cor'
vocab_file = '/home/unity/IVR/ConversationalIvr/utility' + os.sep + 'text8'
                


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def load_model():
    model_name = '/home/unity/IVR/ConversationalIvr/utility/gensim/trained_model_doc2vec'
    model = gensim.models.doc2vec.Doc2Vec.load(model_name)
    return model

def test_model(query):
    ranks = []
    second_ranks = []
    query = model.infer_vector(query.split(" "))
    #doc_id = random.randint(0, len(train_corpus) - 1)
    #inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([query], topn=len(model.docvecs))

    #rank = [docid for docid, sim in sims].index(doc_id)
    #ranks.append(rank)

    #second_ranks.append(sims[1])

    print('Question ({}): «{}»\n'.format(sims[0][0], ' '.join(train_corpus[sims[0][0]].words)))
    print('Answer  ({}): «{}»\n'.format(sims[0][0], ' '.join(train_answer_corpus[sims[0][0]].words)))

    print("Question Score",sims[0][1])
    if sims[0][1] > 0.7:        
        answer = ' '.join(train_answer_corpus[sims[0][0]].words)
        return answer
    else:
        answer = "Sorry i am not very sure what to answer for this query"
        return answer


print("Reading Corpus for doc2vec")
train_corpus = list(read_corpus(lee_train_file))
train_answer_corpus = list(read_corpus(train_answer_file))
print("Reading corpus completed")

model = load_model()


query = "harry potter books worth"
test_model(query)

    
