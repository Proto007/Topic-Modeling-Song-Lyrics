import GarbageRemoval as gr
import DocsProcessing as dp
import LDAModel as lda
import os
import pandas as pd
from pprintpp import pprint
from collections import Counter
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import ldamodel
import matplotlib.pyplot as plt

#Makes a list of all words in all songs.
#@songs_list is a list of lists with each list representing one song
#returns a list of strings
def GetAllWords(songs_list):
  all_words=[]
  for song in songs_list:
    all_words.extend(song)
  return all_words

#Makes a list of all unique words from a list of all songs
#@songs_list is a list of lists with each list representing one song
#returns a list of strings
def GetUniqueWords(songs_list):
    unique = []
    for song in songs_list:
      for word in song:
        if word not in unique:
          unique.append(word)
    return unique

#Calculates type to token ratio: uniquewords/allwords
#@songs_list is a list of lists with each list representing one song
#returns a float
def GetTtr(songs_list):
  return float(len(GetUniqueWords(songs_list))/len(GetAllWords(songs_list)))

#Calculates the average word per song: allwordscount/numberofsongs
#@songs_list is a list of lists with each list representing one song
#returns a float
def AverageWordsPerSong(songs_list):
  return len(GetAllWords(songs_list))/len(songs_list)

#Returns a list of @num_of_words songs from @songs_list
#@songs_list is a list of lists with each list representing one song
#@num_of_words is an integer
#returns a list of strings
def TopWords(songs_list,num_of_words):
  counter = Counter(GetAllWords(songs_list))
  return counter.most_common(num_of_words)

def VisualizeLda(lda_model,corpus,dictionary,html_name):
  LDAvis_prepared = gensimvis.prepare(lda_model, corpus, dictionary)
  if(not os.path.exists(os.getcwd()+"/Model Visualization")):
    os.mkdir(os.getcwd()+"/Model Visualization")
  pyLDAvis.save_html(LDAvis_prepared,os.getcwd()+"/Model Visualization/"+html_name)
  #pyLDAvis.show(LDAvis_prepared)

  
#Calculates perplexity score of the given LDA model 
#Lower score means a better model
#@lda_model is an lda model
#@corpus is the sparse matrix created for the @lda_model
#returns a float
def GetPerplexity(lda_model, corpus):
  return lda_model.log_perplexity(corpus)

#u_mass and c_v
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

def GetTopicModels(corpus, id2word, songs_list, max_topics):
  coherence_values=[]
  lda_models=[]
  perplexity_list=[]
  for i in range(2,max_topics+1,2):
    a_model=ldamodel.LdaModel( corpus = corpus, id2word = id2word, num_topics = i, random_state = 100, update_every = 1, passes = 10,
                                 alpha = 'auto', per_word_topics = True) 
    lda_models.append(a_model)
    coherence_values.append(GetCoherence(a_model,corpus,songs_list,id2word,"u_mass"))
    lda.SaveToDisk(a_model, str(i)+" Topics Model Test")
    perplexity_list.append(GetPerplexity(a_model,corpus))
  return lda_models,coherence_values,perplexity_list

def GetCoherenceGraph(coherence_values,max_topics):
  x=range(2,max_topics+1,2)
  plt.plot(x,coherence_values,label="Coherence Score")
  plt.xlabel("Number of Topics")
  plt.legend(("coherence_values"), loc='best')
  plt.savefig("Topic Coherence.png")
  plt.show()

def GetPerplexityGraph(perplexity_list,max_topics):
  x=range(2,max_topics+1,2)
  plt.plot(x,perplexity_list,label="Perplexity")
  plt.xlabel("Number of Topics")
  plt.legend(("perplexity_list"), loc='best')
  plt.savefig("Topic Perplexity.png")
  plt.show()

def ReadTestData(data_type,folder_name):
  directory=os.getcwd()+'/'+folder_name
  cwd=os.getcwd()
  os.chdir(directory)
  path=os.getcwd()+"/"+data_type
  songs_list=[]
  songs_name_list=[]
  for root, subdir, file_names in os.walk(path):
    for file in file_names:
      with open(os.path.join(root, file),'r',encoding='utf8') as file_data:
        song=file_data.read()
        song=dp.PreprocessDocs(song,0)
        if len(song)>0:
          songs_list.append(song)
          songs_name_list.append(str(file))
  os.chdir(cwd)
  return songs_list,songs_name_list


def TestToCsv(lda_model,corpus,model_id2word,songs_list,songs_name_list,num_of_topics,file_name):
  all_topics=[]
  for i in range(num_of_topics):
    all_topics.append([])
  for data in lda_model.get_document_topics(corpus,minimum_probability=0):
    for i in range(num_of_topics):
      all_topics[i].append(round(data[i][1],5))
  dominant_topics=[]
  dominant_topic_words=[]
  for i in range(len(songs_list)):
    highest_prob=0.0
    highest_index=0
    for j in range(len(all_topics)):
      if(all_topics[j][i]>highest_prob):
        highest_prob=all_topics[j][i]
        highest_index=j
    topic_words=lda_model.get_topic_terms(highest_index,20);
    dominant_topic_words.append([model_id2word[word[0]] for word in topic_words])
    dominant_topics.append("Topic "+str(highest_index))
  df_data={"File Name":songs_name_list, "Bag Of Words":songs_list,"Dominant Topic":dominant_topics,"Dominant Topic Words":dominant_topic_words}
  for i in range(num_of_topics):
    df_data["Topic "+str(i)]=all_topics[i]
  topics_df=pd.DataFrame(df_data)
  pprint(topics_df)
  topics_df.to_csv(os.getcwd()+"/"+file_name,index=False)

def GetTestInfo(file_name, num_of_topics,lda_model,model_id2word):
  df=pd.read_csv(file_name)
  song_count=df["Dominant Topic"].count()
  topic_groups=df.groupby("Dominant Topic")
  topic_names=[]
  topic_count=[]
  topic_ratio=[]
  dominant_topic_words=[]
  for i in range(num_of_topics):
    topic_names.append("Topic "+str(i))
    topic_words=lda_model.get_topic_terms(i,5);
    dominant_topic_words.append([model_id2word[word[0]] for word in topic_words])
    try:
      topic_count.append(len(topic_groups.get_group("Topic "+str(i))))
      topic_ratio.append(len(topic_groups.get_group("Topic "+str(i)))/song_count)
    except:
      topic_count.append(0)
      topic_ratio.append(0)
  topics_data={"Topic":topic_names,"Count": topic_count,"Ratio":topic_ratio,"Topic Words":dominant_topic_words}
  topics_df=pd.DataFrame(topics_data)
  #topics_df.to_csv(os.getcwd()+"/"+"Topic Data.csv")
  return topics_df

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
      file_names.append(labeled_info[0].strip())
      human_labels=labeled_info[1].split(",")
      label_1.append(human_labels[0].strip())
      label_2.append(human_labels[1].strip())
      label_3.append(human_labels[2].strip())
    label_data={"File Name":file_names,"Human Label 1":label_1,"Human Label 2":label_2,"Human Label 3":label_3}
    label_df=pd.DataFrame(label_data)
    label_df.to_csv("Labels.csv")
    return label_df
  
def CheckSimilarity(full_csv, chorus_csv, labeled_csv):
  full_df=pd.read_csv(full_csv)
  chorus_df=pd.read_csv(chorus_csv)
  labeled_df=pd.read_csv(labeled_csv)
  name_list=[]
  chorus_topic_list=[]
  full_topic_list=[]
  human_label=[]
  human_chorus_agreement=[]
  human_full_agreement=[]
  chorus_full_agreement=[]
  for row,name in enumerate(labeled_df["File Name"]):
    name_list.append(name)
    chorus_index= chorus_df.index[chorus_df["File Name"]==name.strip()].tolist()[0]
    full_index= full_df.index[full_df["File Name"]==name.strip()].tolist()[0]
    chorus_topic=chorus_df["Dominant Topic"][chorus_index]
    full_topic=full_df["Dominant Topic"][full_index]
    chorus_topic_list.append(chorus_topic)
    full_topic_list.append(full_topic)
    human_label_1=str(labeled_df["Human Label 1"][row])
    human_label_2=str(labeled_df["Human Label 2"][row])
    human_label_3=str(labeled_df["Human Label 3"][row])

    if(human_label_1==chorus_topic or human_label_2==chorus_topic or human_label_3==chorus_topic):
      human_chorus_agreement.append(1)
    else:
      human_chorus_agreement.append(0)
    if(human_label_1==full_topic or human_label_2==full_topic or human_label_3==full_topic):
      human_full_agreement.append(1)
    else:
      human_full_agreement.append(0)
    if(full_topic==chorus_topic):
      chorus_full_agreement.append(1)
    else:
      chorus_full_agreement.append(0)
    human_label.append(human_label_1+", "+human_label_2+", "+human_label_3)
  similarity_data={"File Names":name_list,"Chorus":chorus_topic_list,
                   "Full":full_topic_list,"Human Label":human_label,
                   "Human-chorus agreement":human_chorus_agreement,
                   "Human-full agreement":human_full_agreement,
                   "Full-Chorus agreement":chorus_full_agreement}
  similarity_df=pd.DataFrame(similarity_data)
  similarity_df.to_csv("Similarity.csv")
  print("Human chorus agreement:",sum(human_chorus_agreement)/len(human_chorus_agreement))
  print("Human fullsong agreement:",sum(human_full_agreement)/len(human_full_agreement))
  print("Chorus fullsong agreement:",sum(chorus_full_agreement)/len(chorus_full_agreement))
  return similarity_df
