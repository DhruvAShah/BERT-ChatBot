import gensim
import os
import collections
import smart_open
import random
import spacy
nlp = spacy.load('en_core_web_lg')

# Set file names for train and test data
#test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
test_data_dir = '/home/unity/IVR/ConversationalIvr/utility/gensim_unity'
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

def remove_s(sent):
    sent_new = []
    for word in sent:
        if word[-1] == 's':
            sent_new.append(word[0:-1])
        else:
            sent_new.append(word)
    return sent_new

def train_model():
    model = gensim.models.doc2vec.Doc2Vec(vector_size=1024, min_count=1, epochs=1000, window=1, dbow_words=1, dm=1)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("doc2vec_unity_new")
    return model

def load_model():
    model_name = '/home/unity/IVR/ConversationalIvr/utility/gensim_unity/doc2vec_unity_new'
    model = gensim.models.doc2vec.Doc2Vec.load(model_name)
    return model

def test_model_unity(query1):
    ranks = []
    second_ranks = []
    query = model.infer_vector(query1.split(" "))
    sims = model.docvecs.most_similar([query], topn=len(model.docvecs))

    #print('Question ({}): «{}»\n'.format(sims[0][0], ' '.join(train_corpus[sims[0][0]].words)))
    #print('Answer  ({}): «{}»\n'.format(sims[0][0], ' '.join(train_answer_corpus[sims[0][0]].words)))
    #print("Question Score",sims[0][1])  
    
    length_query1 = len(query1.split(" "))
    
    query1 = nlp(query1)
    
    
    entity = []
    
    for token in query1:
        #print(token.text, token.pos_, token.dep_)
        if token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
            entity.append(token.text.lower())    

    entity = remove_s(entity)
    
    print(entity)
    best = (-1,-1000,-1)

    
    for i in range(10):
        print("Question ", train_corpus[sims[i][0]])
        print("Score ", sims[i][1])
        
        score = 0 
        for w in entity:
            if ' '.join(train_corpus[sims[i][0]].words).lower().find(w.lower()) > -1:
                score += 5
        score -= abs(length_query1 - len(train_corpus[sims[i][0]].words))
        if score > best[1]:
            best = (sims[i][0], score, sims[i][1])
            
    
    if best[2] > 0.4:        
        answer = ' '.join(train_answer_corpus[best[0]].words)
        print("Final Question ", ' '.join(train_corpus[best[0]].words))
        return answer
    else:
        answer = "Sorry i am not very sure what to answer for this query"
        return answer

print("Reading Corpus for doc2vec")
train_corpus = list(read_corpus(lee_train_file))
train_answer_corpus = list(read_corpus(train_answer_file))
print("Reading corpus completed")

model = load_model()

#print("Training model")
#model = train_model()
#print("Training Completed")

#query = "what number of max ports in cuc can be configured"
#test_model_unity(query)
