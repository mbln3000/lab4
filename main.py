# 5. Научиться работать с библиотекой natasha. Задачи:
# i.	Загрузить текстовые данные (не менее 2000 символов)
# ii.	Разделить текст на предложения
# iii.	Выделить токены и провести частеречную разметку, вывести на экран первые 20 токенов с тэгами
# iv.	Нормализовать именованные сущности в тексте
# v.	Выделить даты и вывести их в формате число-месяц-год

import natasha
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    PER,
    NamesExtractor,
    DatesExtractor,
    Doc
)
import codecs


# segmenter = Segmenter()
# morph_vocab = MorphVocab()
# emb = NewsEmbedding()
# morph_tagger = NewsMorphTagger(emb)
# syntax_parser = NewsSyntaxParser(emb)
# ner_tagger = NewsNERTagger(emb)
# names_extractor = NamesExtractor(morph_vocab)
# dates_extractor = DatesExtractor(morph_vocab)
#
#
# with codecs.open("monkeys.txt", encoding='utf-8') as file:
#     text = file.read()
#
# doc = Doc(text)
#
# doc.segment(segmenter)
# tokens = doc.tokens #деление на токены
# sentences = doc.sents #деление на предложения
#
# doc.tag_morph(morph_tagger)
# print(doc.tokens[:20]) #первые 20 токенов с частеречной разметкой
#
# doc.tag_ner(ner_tagger)
# for span in doc.spans:
#     print(span)
#     span.normalize(morph_vocab)
# ners = doc.spans #именованные сущности
#
# dates = list(dates_extractor(text))
# print(dates) #даты

# 6. Средствами NLTK выделить именованные сущности с тэгами (Person, Organisation, GSP и проч.)
# для английского и русского текста.

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
nltk.download('maxent_ne_chunker')

# #Для текста на русском
# with codecs.open("rus.txt", encoding='utf-8') as file:
#     text_rus = file.read()
#
# sentences = text_rus
# tokens = nltk.word_tokenize(sentences)
# tagged = nltk.pos_tag(tokens)
# entities = nltk.chunk.ne_chunk(tagged)
#
# for sent in nltk.sent_tokenize(text_rus):
#    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
#       if hasattr(chunk, 'label'):
#          print(chunk.label(), ' '.join(c[0] for c in chunk))
#
# #Для текста на английском:
#
# with codecs.open("eng.txt", encoding='utf-8') as file:
#     text_eng = file.read()
#
# sentences = text_eng
# tokens = nltk.word_tokenize(sentences)
# tagged = nltk.pos_tag(tokens)
# entities = nltk.chunk.ne_chunk(tagged)
#
# for sent in nltk.sent_tokenize(text_eng):
#    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
#       if hasattr(chunk, 'label'):
#          print(chunk.label(), ' '.join(c[0] for c in chunk))


# 7. С помощью sklearn обучить модель распознавать части речи в предложении.
# Для этого необходимо разбить данные на обучающую и тестовую выборки,
# а в конце вывести на экран предсказание модели и степень его точности.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk import word_tokenize, pos_tag

tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


import pprint

pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))

{'capitals_inside': False,
 'has_hyphen': False,
 'is_all_caps': False,
 'is_all_lower': True,
 'is_capitalized': False,
 'is_first': False,
 'is_last': False,
 'is_numeric': False,
 'next_word': 'sentence',
 'prefix-1': 'a',
 'prefix-2': 'a',
 'prefix-3': 'a',
 'prev_word': 'is',
 'suffix-1': 'a',
 'suffix-2': 'a',
 'suffix-3': 'a',
 'word': 'a'}

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


cutoff = int(.75 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]

print(len(training_sentences))
print(len(test_sentences))


def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])

    return X, y

X, y = transform_to_dataset(training_sentences)

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X[:10000],
        y[:10000])  # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print('Training completed')

X_test, y_test = transform_to_dataset(test_sentences)

print("Accuracy:", clf.score(X_test, y_test))


def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return [i for i in zip(sentence, tags)]

print(pos_tag(word_tokenize('As they moved through the water, Charles seemed enraptured by ice stalactites and tiny amphipods, later commenting on his love for the “sacred qualities” of the natural world.')))

