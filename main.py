import GarbageRemoval as gr
import DocsProcessing as dp
import LDAModel as lda
import Findings as fd
import os


###Creating BOW for "Music4all"###
"""
songs_list=dp.ReadFolder("Music4all",20,True,20)
songs_list=[] #Can be left empty for "u_mass" coherence
"""

###Creating an LDA model from "Music4all" folder###
#lda_model=lda.MakeLdaModel("Music4all",False,20,20,True,True,20)

###Saving the created LDA model in disk###
#lda.SaveToDisk(lda_model,"Bigrams 20 Topics Final")

###Loading the model and corresponding corpus/Dictionary from disk###

lda_model=lda.LoadFromDisk("Bigrams 20 Topics Final")
corpus=lda.LoadCorpus()
id2word=lda.LoadDictionary()

###Creating a visualization for the model and printing the topics###
"""
fd.VisualizeLda(lda_model,corpus,id2word,"Bigrams 20 Topics Final.html")
pprint(lda_model.print_topics(20,20))
"""
###Make CSV file with the topics and words###
"""
fd.GetTopicWordsCsv(lda_model,id2word,20)
"""
###Getting information about the corpus###
"""
print("Total Words: ",fd.GetTotalWords(corpus))
print("Total Unique Words: ",len(fd.GetUniqueWords(id2word)))
print("Type Token Ratio: ",fd.GetTtr(corpus,id2word))
print("Average Words Per Song: ",fd.AverageWordsPerSong(corpus))
print("Top 10 Words: ",fd.TopWords(id2word,10))
"""
###Getting model Perplexity and Coherence###
"""
print("Per: ",fd.GetPerplexity(lda_model,corpus))
print("Coh: ",fd.GetCoherence(lda_model,corpus,songs_list,id2word,"u_mass"))
"""
###Making 50 models to get a perplexity and Coherence Graph###
"""
lda_models,coherence_vals,perplexities=fd.GetTopicModels(corpus,id2word,songs_list,50)
fd.GetCoherenceGraph(coherence_vals,50)
fd.GetPerplexityGraph(perplexities,50)

print("Coherences: ",coherence_vals)
print("Perplexities: ",perplexities)
"""
###Testing and evaluation###

chorus_list,ch_name_list=fd.ReadTestData("Chorus",True)
full_list,fs_name_list=fd.ReadTestData("Full",True)

fd.TestToCsv(lda_model, corpus, chorus_list,ch_name_list,20,"chorus_topics.csv")
fd.TestToCsv(lda_model, corpus, full_list,fs_name_list,20,"full_topics.csv")

fd.GetTestInfo("chorus_topics.csv", 20,lda_model,id2word, "chorus_topics_info.csv")
fd.GetTestInfo("full_topics.csv", 20,lda_model,id2word, "full_topics_info.csv")

fd.GetLabeledCsv("labels.txt")

fd.CheckSimilarity("full_topics.csv","chorus_topics.csv","labels.csv")

