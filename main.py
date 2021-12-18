import GarbageRemoval as gr
import DocsProcessing as dp
import LDAModel as lda
import Findings as fd
from pprintpp import pprint
import os

#songs_list=dp.ReadFolder("Music4all",5,True)
#id2word=lda.GetDictionary(songs_list,True)
#corpus=lda.GetCorpus(id2word,songs_list)


"""
coherence_vals=[]
perplexities=[]
for i in range(2,101,2):
    lda_model=lda.LoadFromDisk(str(i)+" Topics Model Test")
    try:
        fd.VisualizeLda(lda_model,corpus,id2word,str(i)+" Topics Bigram.html")
    except:
        print(i)
        continue
"""



corpus=lda.LoadCorpus()
id2word=lda.LoadDictionary()
#lda_model=lda.MakeLdaModel("Music4all",True,15,5,True,True)
#lda.SaveToDisk(lda_model,"Bigrams 15 topics")
#print(dp.ReadFolder("Testing",1,True))
#print("Per: ",fd.GetPerplexity(lda_model,corpus))
#print("Coh: ",fd.GetCoherence(lda_model,corpus,songs_list,id2word,"u_mass"))
lda_model=lda.LoadFromDisk("Bigrams 15 topics")
fd.GetTopicWordsCsv(lda_model,id2word,15)
#lda_models,coherence_vals,perplexities=fd.GetTopicModels(corpus,id2word,songs_list,21)

#fd.GetCoherenceGraph(coherence_vals,100)
#fd.GetPerplexityGraph(perplexities,100)
#print("Coherences: ",coherence_vals)
#print("Perplexities: ",perplexities)
#pprint(lda_model.print_topics(20,30))
#fd.VisualizeLda(lda_model,corpus,id2word,"Bigrams 15 topics.html")

'''
test_songs,names=fd.ReadTestData("Full","Test Folder")
test_id2word=lda.GetDictionary(test_songs,False)
test_corpus=lda.GetCorpus(test_id2word,test_songs)
fd.TestToCsv(lda_model,test_corpus,id2word,test_songs,names,10,"Full.csv")
'''

"""
Per:  -8.892340630862416
Coh:  -3.6179068155906218
"""
