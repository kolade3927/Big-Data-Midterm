#!/usr/bin/env python
# coding: utf-8

# Instructions
# 
# For this midterm you will be performing an analysis on a document that has been converted using a
# Caesar Cypher.
# 
# To do this you will do the following:
# 
# - Word Count of the document
# - Character Count of the document
# - Comparison to the expected frequencies of characters in the English Language
# - Use a natural language processing library to compare samples of the words within your
# document to see if they are valid english words
# - Find the correct decryption of the document and save the output file.

import os
import string
from operator import add

import nltk 
from nltk import word_tokenize
from nltk.corpus import words

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row, functions as sql_f


# Set up environment configurations
conf = SparkConf().setAppName('digits-recognition')
conf.set('spark.executor.memory', '12g') # 12 Gigabytes of RAM memory

# Create a spark SQL session, or get if already existing (only one instance)
sess = SparkSession.builder.config(conf = conf).getOrCreate()

# Reduce the verbose log
#sess.sparkContext.setLogLevel("WARN")
print(sess)

# Read the text file, and cache
filename = 'Encrypted-2.txt'
df = sess.read.text(filename)
df.cache()

df.show(10)
print('Total lines:', df.count())

# Convert the Dataframe to RDD
rdd = df.rdd.map(lambda row: row.value)

# ### Word and Character Counts of the document

def remove_punctuations(text):
    punctuations = string.punctuation
    trans = str.maketrans('', '', punctuations)
    
    return text.translate(trans)


def split_characters(text, remove_num=True):
    # Remove spaces
    text = text.replace(' ', '')
    
    # Split into characters
    chars = list(text)
    
    if remove_num:
        # Remove numerical characters
        chars = [c for c in chars if c.isalpha()]
    
    return chars


def count_entities(rdd, level='word', remove_punc=True):
    if remove_punc:
        # Remove punctuations
        rdd = rdd.map(remove_punctuations)
        print('\nAfter removing punctuations:')
        print( rdd.take(5) )
    
    # Split text into entities
    if level == 'word':
        rdd_entities = rdd.flatMap(lambda line: line.split(' ') )
    else:
        rdd_entities = rdd.flatMap(split_characters)
        
    print('\nAfter spliting into entities:')
    print( rdd_entities.take(5) )
    
    # Prepare each entity for counting
    rdd_entities_once = rdd_entities.map(lambda entity: (entity, 1))
    print('\nEach time an entity occurs:')
    print( rdd_entities_once.take(5) )
    
    # Aggregate count all the entities occurences
    rdd_entity_counts = rdd_entities_once.reduceByKey(add)
    print('\nEntity counts:')
    print( rdd_entity_counts.take(5) )
    
    return rdd_entity_counts


# #### Word counts
rdd_word_counts = count_entities(rdd, level='word', remove_punc=True)

# #### Character counts
rdd_char_counts = count_entities(rdd, level='character', remove_punc=True)

# ##### Expecting maximum of 26
print('Number of characters:', rdd_char_counts.count())

# #### Relative frequencies
total_chars = rdd_char_counts.map(lambda c: c[1]).sum()
print('Total characters in the text:', total_chars)


def rel_freq(char, total):
    rel_freq = 100* (char[1] / total)
    
    return char[0], round(rel_freq, 2)


rdd_rel_freq = rdd_char_counts.map(lambda char: rel_freq(char, total_chars) )
rdd_rel_freq = rdd_rel_freq.sortByKey()

print('Text Character Relative Frequencies:')
print( rdd_rel_freq.collect() )


# ### Comparison to the expected frequencies of characters in the English Language
# 
# Load the Letter frequency from Wikipedia [src: https://en.wikipedia.org/wiki/Letter_frequency ]


character_freq_filename = 'character_frequencies.txt'
df_char_freq = sess.read.csv(character_freq_filename,
                              header=True,
                              sep='\t')

print('Columns:', df_char_freq.columns)
df_char_freq.select('Letter', 'English').show()


# Convert df to rdd
rdd_char_freq = df_char_freq.rdd


def parse_freq(val):
    val = val.replace('%', '')
    return float(val)


# Extract the English character values
rdd_en_char_freq = rdd_char_freq.map(lambda row: (row.Letter.upper(), parse_freq(row.English)))

print('Expected English Character Frequencies:')
print( rdd_en_char_freq.collect() )


# #### Comparison summary

rdd_comparison = rdd_en_char_freq.join(rdd_rel_freq)
rdd_comparison = rdd_comparison.map(lambda r: Row(letter=r[0],
                                                  expected=r[1][0],
                                                  found=r[1][0]
                                                 ))

print('Character frequencies summary:')
print(rdd_comparison.collect())

rdd_comparison.toDF().select('letter', 'expected', 'found').show()


# ### Use a natural language processing library to compare samples of the words within your document to see if they are valid english words

english_words = set( words.words() )

def is_english_word(word):
    return word.lower() in english_words


def validate_english_words(rdd):
    # Tokenize 
    rdd_clean = rdd.map(remove_punctuations)
    rdd_tokens = rdd_clean.flatMap(word_tokenize)
    print('\nTokenized words:')
    print( rdd_tokens.take(10) )
    
    # Check for English words
    rdd_english_tokens = rdd_tokens.map(lambda token: (token, is_english_word(token)) )
    print('\nEnglish words validation:')
    print( rdd_english_tokens.take(10) )
    
    return rdd_english_tokens


# #### Take some samples to test with

n_samples = 5
rdd_sample = df.limit(n_samples).rdd.map(lambda row: row.value)

print('Sample text:')
print( rdd_sample.collect() )

validate_english_words(rdd_sample)
# They are not english words


# ### Find the correct decryption of the document and save the output file.

letters = string.ascii_uppercase

def decrypt(text, shift):
    decrypted_text = ''
    
    for ch in text:
        if ch.isalpha():
            index = (letters.index(ch) - shift) % 26
            decrypted_text += letters[index]
        else:
            decrypted_text += ch
            
    return decrypted_text


## Testing the decrypt function
cipher = 'QEB NRFZH YOLTK CLU GRJMP LSBO QEB IXWV ALD'
print(cipher, 'deciphered to:', decrypt(cipher, 23))


### Search for the optimal shift
valid_shift = None
for n in range(26):
    # Attempt to decrypt with shift n
    rdd_decrypted = rdd_sample.map(lambda line: decrypt(line, n) )
    
    print('Decryption with shift:', n)
    print( rdd_decrypted.collect() )
    
    # Validate English words
    rdd_english_words = validate_english_words(rdd_decrypted)
    
    # Count the number of valid English words
    english_words_count = rdd_english_words.map(lambda token: token[1]).reduce(add)
    
    # Proportion of valid English words
    fraction = english_words_count / rdd_english_words.count()
    print('Fractional valid English word:', fraction)
    
    if fraction > 0.5:
        valid_shift = n
        print('\nHurray! the correct shift is:', valid_shift)
        break
    
    print('\n\n')


# #### Saving the decrypted text to file
print('Valid shift:', valid_shift)

rdd_decrypted = rdd.map(lambda line: decrypt(line, valid_shift))

result_filename = 'D'+filename
if not os.path.exists(result_filename):
    rdd_decrypted.saveAsTextFile(result_filename)

