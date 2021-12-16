#Sadab Hafiz and Zachary Motassim
#This file contains functions to make LDA model and all it's prerequisites. Functions are provided to save and load an LDA model from disk.

#Import the necessary modules and implementation files
import DocsProcessing as dp
import os
import gensim
from gensim import corpora
from gensim.models import ldamodel
from gensim.models import CoherenceModel # import for coherence model

#Takes a dictionary representation of the songs along with the bag of words list for each song and applies the id2words on each song
#Corpus is a requirement for making the LDA model
#@id2word is a dictionary object created through gensim's corpora.Dictionary() function
#@songs_list is a list of lists where each sublist is a bag of word for each song
#returns corpus usable by lda model
def GetCorpus(id2word,songs_list):
  corpus = [id2word.doc2bow(song) for song in songs_list]
  return corpus

#Takes a list of bag of words for each song and makes a dictionary using gensim's id2word
#Each word is provided its unique id
#@filter_words is a bool which determines whether or not certain words get filtered out based on frequency
#@songs_list is a list of lists where each sublist is a bag of word for each song
#id2word dictionary is a requirement for the LDA model
#returns a dictionary with each words having a unique id
def GetDictionary(songs_list,filter_words):
  id2word = corpora.Dictionary(songs_list) # make dictionary for model
  if filter_words:
    #filter words based on their document frequency
    id2word.filter_extremes(no_below=10, no_above=0.6, keep_tokens=None)
  return id2word

#Makes an LDA model with provided corpora_folder_name, num_of_topics to generate and min_words in each corpus
#Calls functions previously defined to create corpus, dictionary and bag of words list for the model
#@corpora_folder is a string
#@num_of_topics is an integer greater than 0
#@min_words is an integer greater than 0
#@filter_words is a bool to decide whether words are filtered based on frequency
#returns an LDA model with @num_of_topics topics
def MakeLdaModel(corpora_folder,num_of_topics,min_words,filter_words):
  songs_list=dp.ReadFolder(corpora_folder,min_words)
  id2word=GetDictionary(songs_list,filter_words)
  corpus=GetCorpus(id2word,songs_list)
  lda_model = ldamodel.LdaModel( corpus = corpus, id2word = id2word,
                                 num_topics = num_of_topics, random_state = 100,
                                 update_every = 1, passes = 10,
                                 alpha = 'auto', per_word_topics = False)  
  return lda_model

#Save an LDA model in current_working_directory/Saved Models/@name_of_model/
#@name_of_model is a string that determines the file name of the model that is saved and also the folder name
#@lda_model is an lda model
def SaveToDisk(lda_model,name_of_model):
  directory=os.getcwd()+"/Saved Models/"+name_of_model
  if(not os.path.exists(directory)):
    os.mkdir(directory)
  lda_model.save(directory+"/"+name_of_model)

#Takes the @name_of_model and loads LDA model with corresponding name from current working directory/Saved Models
#@name_of_model is a string
#returns an lda model loaded from the disk
def LoadFromDisk(name_of_model):
  file_path=os.getcwd()+"/Saved Models/"+name_of_model+"/"+name_of_model
  lda_model=ldamodel.LdaModel.load(file_path,mmap='r')
  return lda_model




























