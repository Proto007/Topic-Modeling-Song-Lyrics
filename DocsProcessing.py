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

#Takes a list of strings and returns a bigram model for those strings using Gensim's Phrases 
#@all_sentences is a list of strings where each string is a line
#@min_count is an integer that decides how many times a bigram needs to happen for it to be considered
#returns a bigram_model
def GetBigrams(all_sentences,min_count):
  bigram=gensim.models.Phrases(all_sentences,min_count=min_count,threshold=10)
  bigram_model=gensim.models.phrases.Phraser(bigram)
  return bigram_model

#Uses regular expression tokenizer to tokenize a song
#@song is a string 
#returns a list of strings with the @song tokenized into words. Counts words with " ' "
def RegexTokenize(song):
  tokenizer=RegexpTokenizer( '\w+[\']?\w*|[^\W\s]+', gaps = False )
  return tokenizer.tokenize(song)

#Tokenizes a song using NLTK's word_tokenize
#@song is a string
#returns a list of strings with @song tokenized into words
def NltkTokenize(song):
  return word_tokenize(song)

#Removes stopwords that are in nltk's stopwords list and meaningless words that we've encountered from a song
#@song is a list of strings
#returns a list of strings that with only meaningful words
def RemoveStopwords(song):
    stop_words = stopwords.words('english')
    #Stopwords list extended with words we've encountered in the files that don't have any meaning for the LDA model
    stop_words.extend(['ooh',"oooh","gon","wan","...","mmm","nanana"])
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
#returns a list of strings with the song(tokenized, no endline chars, no stopwords, no numbers, all lower case, lemmatized)
#returns empty list for songs that are less than @min_words long.
def PreprocessDocs(song, min_words):
  #song=NltkTokenize(song)
  song=RegexTokenize(song)
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
#@bigrams is a bool that decides whether or not the songs will be read as bigrams with min_count 10
#returns a list of lists where each sublist represents a song
def ReadFolder(folder_name, min_words,bigrams,bigrams_min_count):
  path=os.getcwd()+"/"+folder_name
  songs_list=[]
  song_lines=[]
  for root, subdir, file_names in os.walk(path):
    for file in file_names:
      with open(os.path.join(root, file),'r',encoding='utf8') as file_data:
        if bigrams==True:
          song=file_data.readlines()
          song_bow=[]
          for i in range(len(song)):
            song[i]=PreprocessDocs(song[i],1)
            song_bow.extend(song[i])
          if len(song)>0:
            song_lines.extend(song)
            songs_list.append(song_bow)
        else:
          song=file_data.read()
          song=PreprocessDocs(song,min_words)
          if len(song)>0:
            songs_list.append(song)
  if(bigrams==True):
    bigram_model=GetBigrams(song_lines,bigrams_min_count)
    for i in range(len(songs_list)):
        songs_list[i]=bigram_model[songs_list[i]]
  return songs_list

