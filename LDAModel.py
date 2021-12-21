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
#returns corpus usable by lda model and saves the corpus in "current_working_directory/Saved Corpus" if @save is true
def GetCorpus(id2word,songs_list, save):
  corpus = [id2word.doc2bow(song) for song in songs_list]
  if(save):
    if(not os.path.exists("Saved Corpus")):
      os.mkdir("Saved Corpus")
    corpora.MmCorpus.serialize(os.getcwd()+"/Saved Corpus/Corpus",corpus)
  return corpus

#Loads and returns the saved corpus from "current_working_directory/Saved Corpus"
def LoadCorpus():
  try:
    return corpora.MmCorpus(os.getcwd()+"/Saved Corpus/Corpus")
  except:
    print("File not found.")

#Takes a list of bag of words for each song and makes a dictionary using Gensim's corpora.Dictionary()
#Each word is provided its unique id and frequency
#@filter_words is a bool which determines whether or not certain words get filtered out based on frequency
#If filter_words is true, words that appear in less than 3 songs and more than 70 percent songs get filtered out
#@songs_list is a list of lists where each sublist is a bag of word for each song
#returns a dictionary with each words having a unique id and saves the dictionary in "current_working_directory/Saved Dictionary" if @save is true
def GetDictionary(songs_list,filter_words, save):
  id2word = corpora.Dictionary(songs_list) # make dictionary for model
  if filter_words:
    #filter words based on their document frequency
    id2word.filter_extremes(no_below=3, no_above=0.6, keep_tokens=None)
  if(save):
    if(not os.path.exists("Saved Dictionary")):
      os.mkdir("Saved Dictionary")
    id2word.save(os.getcwd()+"/Saved Dictionary/Dictionary")
  return id2word

#Loads and returns the saved id2word from "current_working_directory/Saved Dictionary"
def LoadDictionary():
  try:
    return corpora.Dictionary.load(os.getcwd()+"/Saved Dictionary/Dictionary")
  except:
    print("File not found.")

#Makes an LDA model with provided corpora_folder_name, num_of_topics to generate and min_words in each corpus
#Calls functions previously defined to create corpus, dictionary and bag of words list for the model
#@corpora_folder is a string
#@is_saved is a bool to decide if user wants to load corpus and id2word from the disk
#@num_of_topics is an integer greater than 0
#@min_words is an integer greater than 0
#@filter_words is a bool to decide whether words are filtered based on frequency
#@bigrams is a bool to decide if songs will be read as bigrams
#@bigrams_min_count is an integer that decides the min_count for bigrams
#returns an LDA model with @num_of_topics topics
def MakeLdaModel(corpora_folder,is_saved,num_of_topics,min_words,filter_words,bigrams,bigrams_min_count):
  if(is_saved):
    id2word=LoadDictionary()
    corpus=LoadCorpus()
  else:
    songs_list=dp.ReadFolder(corpora_folder,min_words,bigrams,bigrams_min_count)
    id2word=GetDictionary(songs_list,filter_words,True)
    corpus=GetCorpus(id2word,songs_list,True)    
  lda_model = ldamodel.LdaModel( corpus = corpus, id2word = id2word,
                                 num_topics = num_of_topics, random_state = 100,
                                 update_every = 1, passes = 10,
                                 alpha = 'auto', per_word_topics = True)  
  return lda_model

#Save an LDA model in "current_working_directory/Saved Models/@name_of_model/"
#@name_of_model is a string that determines the file name of the model that is saved and also the folder name
#@lda_model is an lda model
def SaveToDisk(lda_model,name_of_model):
  directory=os.getcwd()+"/Saved Models/"+name_of_model
  if(not os.path.exists(directory)):
    os.mkdir(directory)
  lda_model.save(directory+"/"+name_of_model)

#Takes the @name_of_model and loads LDA model with corresponding name from "current_working_directory/Saved Models"
#@name_of_model is a string
#returns an lda model loaded from the disk
def LoadFromDisk(name_of_model):
  file_path=os.getcwd()+"/Saved Models/"+name_of_model+"/"+name_of_model
  lda_model=ldamodel.LdaModel.load(file_path,mmap='r')
  return lda_model




























