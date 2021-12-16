#Sadab Hafiz and Zachary Motassim
#This file contains functions to process each corpus before passing it to the LDA Model

#Import the necessary modules
import os
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
import gensim


def GetBigrams(sentences):
  bigram=gensim.models.Phrases(sentences,min_count=5,threshold=100)
  tokenized_sent=[]
  [tokenized_sent.append(RegexTokenize(sentence)) for sentence in sentences]
  song_words=[]
  [song_words.extend(bigram[sentence]) for sentence in tokenized_sent]
  return song_words

#Uses regular expression tokenizer to tokenize a song
#@song is a string 
#returns a list of strings with the @song tokenized into words
def RegexTokenize(song):
  tokenizer=RegexpTokenizer( '\w+[\']?\w*|[^\W\s]+', gaps = False )
  return tokenizer.tokenize(song)

def NltkTokenize(song):
  return word_tokenize(song)

#Removes stopwords that are in nltk's stopwords list and meaningless words that we've encountered from a song
#@song is a list of strings
#returns a list of strings that with only meaningful words
def RemoveStopwords(song):
    stop_words = stopwords.words('english')
    #Stopwords list extended with words we've encountered in the files that don't have any meaning for the LDA model
    stop_words.extend(['chorus','intro','outro','verse','ooh',"oooh","i'm","i'll","i've",
                       "i'd","n't","'re","'ll","'ve","gon","wan","...","'em",
                       "ooh-ooh","uh-uh","oh-oh","mmm",])
    cleaned_song = []
    #Words that are less than two characters long are removed to prevent meaningless words from affecting the model
    for word in song:
        if (word not in stop_words) and (len(word)>2):
          cleaned_song.append(word)
    return cleaned_song

#Removes '\n' characters from a song
#@song is a list of strings
#returns a list of strings after removing '\n' when encountered
def RemoveEndline(song):
    end_lines_removed = [ ]
    for word in song:
        end_lines_removed.append(word.strip('\n'))
    return end_lines_removed

#Changes all upper-case letters in a song into lower-case
#@song is a list of strings
#returns a list of strings with each word is in lower-case
def CaseFold(song):
    song_lower=[]
    for word in song:
      word_lower=word.lower()
      song_lower.append(word_lower)
    return song_lower

#Lematize the words of a song using WordNetLemmatizer()
#@song is a list of strings
#returns a list of strings with each word is lemmatized
def Lemmatize(song):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in song]        

#Removes numeric characters from one song
#@song is a list of strings
#returns a list of strings with all numbers removed
def RemoveNums(song):
    song_removed_num=[]
    for word in song:
      if word.isnumeric()==False:
         song_removed_num.append(word)
    return song_removed_num

#Calls functions defined in this file on one song to get the corpus ready for LDA Model
#@song is a string with one song
#@min_words is an integer
#returns a list of strings with the song(tokenized, no endline chars, no stopwords, no numbers, all lower case)
#returns empty list for songs that are less than @min_words long.
def PreprocessDocs(song, min_words):
  #song=NltkTokenize(song)
  song=GetBigrams(song)
  song=RemoveEndline(song)
  song=CaseFold(song)
  song=Lemmatize(song)
  song=RemoveStopwords(song)
  song=RemoveNums(song)
  if(len(song)<min_words):
    return []
  return song

#Runs PreprocessDocs function on the songs that are in current_working_directory/folder_name and creates a list of songs
#Gets rid of songs that are less than @min_words long.
#@folder_name is a string
#@min_words is an integer
#returns a list of lists where each list represents a song
def ReadFolder(folder_name, min_words):
  path=os.getcwd()+"/"+folder_name
  file_data_list=[]
  for root, subdir, file_names in os.walk(path):
    for file in file_names:
      with open(os.path.join(root, file),'r',encoding='utf8') as file_data:
        song=file_data.readlines()
        song=PreprocessDocs(song,min_words)
        if len(song)>0:
          file_data_list.append(song)
  return file_data_list

