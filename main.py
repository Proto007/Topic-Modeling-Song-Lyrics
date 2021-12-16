import GarbageRemoval as gr
import DocsProcessing as dp
import LDAModel as lda
import Findings as fd
from pprintpp import pprint
songs_list=dp.ReadFolder("Music4all",30)
id2word=lda.GetDictionary(songs_list,True)
corpus=lda.GetCorpus(id2word,songs_list)

#lda_models,coherence_vals,perplexities=fd.GetTopicModels(corpus,id2word,songs_list,100)
coherence_vals=[]
perplexities=[]
for i in range(2,101,2):
    lda_model=lda.LoadFromDisk(str(i)+" Topics Model Test")
    try:
        fd.VisualizeLda(lda_model,corpus,id2word,str(i)+" Topics Bigram.html")
    except:
        print(i)
        continue
#fd.GetCoherenceGraph(coherence_vals,100)
#fd.GetPerplexityGraph(perplexities,100)
#print("Coherences: ",coherence_vals)
#print("Perplexities: ",perplexities)


#lda_model=lda.MakeLdaModel("Music4all",10,30,True)
#lda.SaveToDisk(lda_model,"Bigram Test")






#lda_model=lda.LoadFromDisk("Bigram Test")
#pprint(lda_model.print_topics(20,10))
#fd.VisualizeLda(lda_model,corpus,id2word,"Bigram Test.html")
'''
test_songs,names=fd.ReadTestData("Full","Test Folder")
test_id2word=lda.GetDictionary(test_songs,False)
test_corpus=lda.GetCorpus(test_id2word,test_songs)
fd.TestToCsv(lda_model,test_corpus,id2word,test_songs,names,10,"Full.csv")
'''




