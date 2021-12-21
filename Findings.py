#Sadab Hafiz and Zachary Motassim
#This file contains functions to evaluate and test the LDA models

#Import the necessary files and libraries
import DocsProcessing as dp
import LDAModel as lda
import os
import pandas as pd
from pprintpp import pprint
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import ldamodel
import matplotlib.pyplot as plt

#Takes a corpus object and returns total words count
#@corpus is a list of bag of words where each word has its own id and frequency
#returns an integer
def GetTotalWords(corpus):
  total=0
  for song in corpus:
    total+=len(song)
  return total

#Makes a list of all unique words from a given id2word dictionary
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#returns a list of strings
def GetUniqueWords(id2word):
    unique = []
    for word_id in id2word:
      unique.append(id2word[word_id])
    return unique

#Calculates type token ratio: unique_words/all_words
#@corpus is a list of bag of words where each word has its own id and frequency
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#returns a float
def GetTtr(corpus,id2word):
  return float(len(id2word)/GetTotalWords(corpus))

#Calculates the average word per song: allwordscount/numberofsongs
#@corpus is a list of bag of words where each word has its own id and frequency
#returns a float
def AverageWordsPerSong(corpus):
  return GetTotalWords(corpus)/len(corpus)

#Returns a list of @num_of_words words from @id2word based on their frequency
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#@num_of_words is an integer that decides how many top words are returned
#returns a list of strings
def TopWords(id2word,num_of_words):
  return id2word.most_common(num_of_words)

#Uses pyLDAvis library's functions to create a visualization for given LDA model
#Fails to make a visualization for certain models
#The topic numbers don't match up with the LDA model itself
#@lda_model is an lda model
#@corpus is a list of bag of words where each word has its own id and frequency
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#@html_name is a string that decides the name of the saved visualization
#saves the visualization in "current_working_directory/Model Visualization" as @html_name
def VisualizeLda(lda_model,corpus,id2word,html_name):
  LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word, sort_topics=False)
  if(not os.path.exists(os.getcwd()+"/Model Visualization")):
    os.mkdir(os.getcwd()+"/Model Visualization")
  try:
    pyLDAvis.save_html(LDAvis_prepared,os.getcwd()+"/Model Visualization/"+html_name)
    #pyLDAvis.show(LDAvis_prepared)
  except:
    print("Failed to create visualization")

  
#Calculates perplexity score of the given LDA model 
#Lower score means a better model
#@lda_model is an lda model
#@corpus is a list of bag of words where each word has its own id and frequency
#returns a float
def GetPerplexity(lda_model, corpus):
  return lda_model.log_perplexity(corpus)

#Calculates the coherence score of given LDA model based on the given @coherence_parameter
#Tested on "c_v"(extremely slow) and "u_mass"(fast)
#@lda_model is an lda_model
#@corpus is a list of bag of words where each word has its own id and frequency
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#@songs_list is a list of songs where each songs is a bag of words
#returns a float
def GetCoherence(lda_model, corpus, songs_list, id2word, coherence_parameter):
  coherence_parameter=coherence_parameter.lower()
  if(coherence_parameter=='c_v'):
    cm = CoherenceModel(model=lda_model,texts=songs_list ,corpus=corpus, dictionary=id2word, coherence="c_v")
  else:
    try:
      cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence=coherence_parameter)
    except:
      print("Coherence parameter invalid.")
      return 0
  coherence = cm.get_coherence()
  return coherence

#Created @max_topics number of lda models with different number of topics each model
#number of topics go from 2,3,4,...,@max_topics
#@corpus is a list of bag of words where each word has its own id and frequency
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#@songs_list is a list of songs where each songs is a bag of words
#returns a list of lda models, list of preplexity(float) and a list of "u_mass" coherence(float)
def GetTopicModels(corpus, id2word, songs_list, max_topics):
  coherence_values=[]
  lda_models=[]
  perplexity_list=[]
  for i in range(2,max_topics+1,1):
    a_model=ldamodel.LdaModel( corpus = corpus, id2word = id2word, num_topics = i, random_state = 100, update_every = 1, passes = 10,
                                 alpha = 'auto', per_word_topics = True) 
    lda_models.append(a_model)
    coherence_values.append(GetCoherence(a_model,corpus,songs_list,id2word,"u_mass"))
    lda.SaveToDisk(a_model, str(i)+" Topics Model Test")
    perplexity_list.append(GetPerplexity(a_model,corpus))
  return lda_models,coherence_values,perplexity_list

#Takes a list of @coherence_values and creates a graph using plt.plot()
#@coherence_values is a list of floats
#@max_topics is an int that decides the x_axis
#saves the image as "Topic Coherence.png" in current_working_directory
def GetCoherenceGraph(coherence_values,max_topics):
  x=range(2,max_topics+1,1)
  plt.plot(x,coherence_values,label="Coherence Score")
  plt.xlabel("Number of Topics")
  plt.legend(("coherence_values"), loc='best')
  plt.savefig("Topic Coherence.png")
  plt.show()

#Takes a list of perplexity and creates a graph using plt.plot()
#@perplexity_list is a list of floats
#@max_topics is an int that decides the x_axis
#saves the image as "Topic Perplexity.png" in current_working_directory
def GetPerplexityGraph(perplexity_list,max_topics):
  x=range(2,max_topics+1,1)
  plt.plot(x,perplexity_list,label="Perplexity")
  plt.xlabel("Number of Topics")
  plt.legend(("perplexity_list"), loc='best')
  plt.savefig("Topic Perplexity.png")
  plt.show()

#Creates a csv file with top 20 words per topic for given LDA model
#@id2word is a dictionary with a unique id for each word and their frequency in the corpus
#@number_of_topics is an int
#Saves the csv as "topic_words.csv" in current_working_directory
def GetTopicWordsCsv(lda_model,id2word,number_of_topics):
  topics=[]
  topic_words=[]
  for i in range(number_of_topics):
    topics.append("Topic "+str(i))
    per_topic_words=lda_model.get_topic_terms(i,20)
    words_list=[]
    [words_list.append(id2word[word[0]]) for word in per_topic_words]
    topic_words.append(words_list)
  topics_data={"Topic":topics,"Words":topic_words}
  topics_df=pd.DataFrame(topics_data)
  topics_df.to_csv("topic_words.csv",index=False)

#Reads songs from "current_working_directory/Test Folder/@data_type"
#This function is made specifically for test data
#@data_type is a string
#@bigrams is a bool which decides whether or not the data is read as bigrams from @bigrams_folder
#@bigrams_folder is a bool
#returns a list of lists where each sublist is a bag of words representation for a song
#returns a list of the strings with names of files read
def ReadTestData(data_type, bigrams, bigrams_folder):
  path=os.getcwd()+"/"+data_type
  songs_list=dp.ReadFolder(data_type,1,bigrams,bigrams_folder,20)
  songs_name_list=[]
  for root, subdir, file_names in os.walk(path):
    for file in file_names:
      songs_name_list.append(str(file))
  return songs_list,songs_name_list

#Gets three dominant topics of each song in @songs_list using provided @lda_model of @num_of_topics topics
#@lda_model is an lda model
#@songs_list is a list of lists where each sublist is a bag of words for a song
#@songs_name_list is a list of strings with file names for the test data
#@num_of_topics is an integer with the number of topics @lda_model has
#@file_name is a string
#returns pandas dataframe with 3 dominant topics per song
#Saves the created dataframe with the name @file_name in current_working_directory
def TestToCsv(lda_model, corpus, songs_list,songs_name_list,num_of_topics,file_name):
  all_topics=[]
  for i in range(num_of_topics):
    all_topics.append([])
  for data in lda_model.get_document_topics(corpus,minimum_probability=0):
    for i in range(num_of_topics):
      all_topics[i].append(round(data[i][1],5))
  dominant_topic_1=[]
  dominant_topic_2=[]
  dominant_topic_3=[]
  for i in range(len(songs_list)):
    topics_prob=[]
    for j in range(len(all_topics)):
      topics_prob.append(all_topics[j][i])
    sorted_prob=sorted(topics_prob,reverse=True)
    dominant_topic_1.append("Topic "+ str(topics_prob.index(sorted_prob[0])))
    dominant_topic_2.append("Topic "+ str(topics_prob.index(sorted_prob[1])))
    dominant_topic_3.append("Topic "+ str(topics_prob.index(sorted_prob[2])))
  df_data={"File Name":songs_name_list, "Bag Of Words":songs_list,"Dominant Topic 1":dominant_topic_1,
           "Dominant Topic 2":dominant_topic_2,"Dominant Topic 3":dominant_topic_3}
  topics_df=pd.DataFrame(df_data)
  topics_df.to_csv(file_name,index=False)
  return topics_df

#Creates a CSV with the ratio of each topic found in test data
#@file_name is a string with name of the csv file which has information about topics for test songs
#@num_of_topics is the number of topics in @lda_model
#@model_id2word is the dictionary used to build @lda_model
#returns a pandas dataframe with the ratio information
#saves a csv named "test_topic_ratio_data.csv" in current_working_directory
def GetTestInfo(file_name, num_of_topics,lda_model,model_id2word, save_name):
  df=pd.read_csv(file_name)
  song_count=df["Dominant Topic 1"].count()
  topic_groups_1=df.groupby("Dominant Topic 1")
  topic_names=[]
  topic_count=[]
  topic_ratio=[]
  dominant_topic_words=[]
  for i in range(num_of_topics):
    topic_names.append("Topic "+str(i))
    topic_words=lda_model.get_topic_terms(i,5);
    dominant_topic_words.append([model_id2word[word[0]] for word in topic_words])
    topic_total=0
    try:
      topic_total+=len(topic_groups_1.get_group("Topic "+str(i)))
    except:
      topic_total+=0
    topic_ratio.append(topic_total/song_count)
    topic_count.append(topic_total)
  topics_data={"Topic":topic_names,"Count": topic_count,"Ratio":topic_ratio,"Topic Words":dominant_topic_words}
  topics_df=pd.DataFrame(topics_data)
  topics_df.to_csv(save_name)
  return topics_df

#Read human labels from text file called @labeled_file
#Each line of @labeled_file is in this format: <name_of_file> - <label 1>, <label 2>, <label 3>
#Each label is the topic picked by annotator for song in name_of_file
#Example label: "Topic 0"
#returns pandas dataframe with label information
#creates a csv called "labels.csv" in current_working_directory with labels data
def GetLabeledCsv(labeled_file):
  file_names=[]
  label_1=[]
  label_2=[]
  label_3=[]
  with open(labeled_file,"r") as file:
    lines=file.readlines()
    for line in lines:
      if len(line)==0:
        continue
      labeled_info=line.split("-")
      if(len(labeled_info)<2):
        break
      file_names.append(labeled_info[0].strip())
      human_labels=labeled_info[1].split(",")
      label_1.append(human_labels[0].strip())
      label_2.append(human_labels[1].strip())
      label_3.append(human_labels[2].strip())
    label_data={"File Name":file_names,"Human Label 1":label_1,"Human Label 2":label_2,"Human Label 3":label_3}
    label_df=pd.DataFrame(label_data)
    label_df.to_csv("labels.csv")
    return label_df
  
#Checks similarity between human labels and model label
#@full_csv is a string with name of a csv file with model topics for test full songs
#@chorus_csv is a string with name of a csv file with model topics for test choruses
#@labeled csv is a string which has name of a csv file with human labels for test songs
#Prints similarities between human, chorus and full
#returns a pandas dataframe with the similarity information
#saves the dataframe as "similarity.csv" in "current_working_directory"
def CheckSimilarity(full_csv, chorus_csv, labeled_csv):
  #Read the CSV files
  full_df=pd.read_csv(full_csv)
  chorus_df=pd.read_csv(chorus_csv)
  labeled_df=pd.read_csv(labeled_csv)
  
  name_list=[]
  chorus_topics_list=[]
  full_topics_list=[]
  human_labels_list=[]
  human_chorus_agreement=[]
  human_full_agreement=[]
  chorus_full_agreement=[]
  chorus_full_dominant_agreement=[]
  for row,name in enumerate(labeled_df["File Name"]):
    name_list.append(name)
    chorus_index= chorus_df.index[chorus_df["File Name"]==name.strip()].tolist()[0]
    
    full_index= full_df.index[full_df["File Name"]==name.strip()].tolist()[0]
    #Read the chorus topics picked by lda model
    chorus_topics=[]  
    chorus_topic_1=chorus_df["Dominant Topic 1"][chorus_index]
    chorus_topic_2=chorus_df["Dominant Topic 2"][chorus_index]
    chorus_topic_3=chorus_df["Dominant Topic 3"][chorus_index]
    chorus_topics.append(chorus_topic_1)
    chorus_topics.append(chorus_topic_2)
    chorus_topics.append(chorus_topic_3)
    #Read full song topics picked by lda model
    full_topics=[]
    full_topic_1=full_df["Dominant Topic 1"][full_index]
    full_topic_2=full_df["Dominant Topic 2"][full_index]
    full_topic_3=full_df["Dominant Topic 3"][full_index]
    full_topics.append(full_topic_1)
    full_topics.append(full_topic_2)
    full_topics.append(full_topic_3)
    #Read human labels
    human_labels=[]
    human_label_1=str(labeled_df["Human Label 1"][row])
    human_label_2=str(labeled_df["Human Label 2"][row])
    human_label_3=str(labeled_df["Human Label 3"][row])
    human_labels.append(human_label_1)
    human_labels.append(human_label_2)
    human_labels.append(human_label_3)
    
    chorus_topics_list.append(str(chorus_topic_1)+", "+str(chorus_topic_2)+", "+str(chorus_topic_3))
    full_topics_list.append(str(full_topic_1)+", "+str(full_topic_2)+", "+str(full_topic_3))
    human_labels_list.append(str(human_label_1)+", "+str(human_label_2)+", "+str(human_label_3))
    
    if((chorus_topic_1 in human_labels) or (chorus_topic_2 in human_labels) or (chorus_topic_3 in human_labels)):
      human_chorus_agreement.append(1)
    else:
      human_chorus_agreement.append(0)
      
    if((full_topic_1 in human_labels) or (full_topic_2 in human_labels) or (full_topic_3 in human_labels)):
      human_full_agreement.append(1)
    else:
      human_full_agreement.append(0)

    if((full_topic_1 in chorus_topics) or (full_topic_2 in chorus_topics) or (full_topic_3 in chorus_topics)):
      chorus_full_agreement.append(1)
    else:
      chorus_full_agreement.append(0)

    if(full_topic_1==chorus_topic_1):
      chorus_full_dominant_agreement.append(1)
    else:
      chorus_full_dominant_agreement.append(0)
      
  #Create the df and CSV    
  similarity_data={"File Names":name_list,"Chorus":chorus_topics_list,
                   "Full":full_topics_list,"Human Label":human_labels_list,
                   "Human-chorus agreement":human_chorus_agreement,
                   "Human-full agreement":human_full_agreement,
                   "Full-Chorus agreement":chorus_full_agreement,
                   "Full-Chorus dominant agreement":chorus_full_dominant_agreement}
  similarity_df=pd.DataFrame(similarity_data)
  similarity_df.to_csv("similarity.csv")
  #Print the agreement probabilities
  print("Human chorus agreement:",sum(human_chorus_agreement)/len(human_chorus_agreement))
  print("Human fullsong agreement:",sum(human_full_agreement)/len(human_full_agreement))
  print("Chorus fullsong agreement:",sum(chorus_full_agreement)/len(chorus_full_agreement))
  print("Chorus fullsong dominant topic agreement:",sum(chorus_full_dominant_agreement)/len(chorus_full_dominant_agreement))
  return similarity_df
