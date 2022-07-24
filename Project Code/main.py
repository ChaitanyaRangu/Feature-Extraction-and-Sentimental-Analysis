# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:51:19 2019

@author: irish/rang
"""


filename="data.txt"
#file = open(filename, 'r') 
#for line in file:
#       print (line)

reviewTitle=[]

reviewContent=[]




with open(filename) as f:
	review = []
	for line in f:
		if line[:6] == "[+][t]":							
			if review:
				reviewContent.append(review)
				review = []
			reviewTitle.append(line.split("[+][t]")[1].rstrip("\r\n"))
			
		elif line[:6] == "[-][t]":
			if review:
				reviewContent.append(review)
				review = []
			reviewTitle.append(line.split("[-][t]")[1].rstrip("\r\n"))
			
		elif line[:6] == "[N][t]":
			if review:
				reviewContent.append(review)
				review = []
			reviewTitle.append(line.split("[N][t]")[1].rstrip("\r\n"))
			
		else:
			if "##" in line:								
				x = line.split("##")
				for i in range(1, len(x)):			#x[0] is the feature given the file.Its been ignored here as its not a part of the review
					review.append(x[i].rstrip("\r\n"))
			else:
				continue
	reviewContent.append(review)
#print (review)
#print (reviewContent)
#print (reviewTitle)

from nltk.corpus import stopwords
import nltk
import string
import re
from collections import OrderedDict
from textblob import TextBlob, Word
from textblob import Blobber
from textblob.taggers import NLTKTagger
import operator
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 


apostropheList = {"n't" : "not","aren't" : "are not","can't" : "cannot","couldn't" : "could not","didn't" : "did not","doesn't" : "does not", \
				  "don't" : "do not","hadn't" : "had not","hasn't" : "has not","haven't" : "have not","he'd" : "he had","he'll" : "he will", \
				  "he's" : "he is","I'd" : "I had","I'll" : "I will","I'm" : "I am","I've" : "I have","isn't" : "is not","it's" : \
				  "it is","let's" : "let us","mustn't" : "must not","shan't" : "shall not","she'd" : "she had","she'll" : "she will", \
				  "she's" : "she is", "shouldn't" : "should not","that's" : "that is","there's" : "there is","they'd" : "they had", \
				  "they'll" : "they will", "they're" : "they are","they've" : "they have","we'd" : "we had","we're" : "we are","we've" : "we have", \
				  "weren't" : "were not", "what'll" : "what will","what're" : "what are","what's" : "what is","what've" : "what have", \
				  "where's" : "where is","who'd" : "who had", "who'll" : "who will","who're" : "who are","who's" : "who is","who've" : "who have", \
				  "won't" : "will not","wouldn't" : "would not", "you'd" : "you had","you'll" : "you will","you're" : "you are","you've" : "you have"}

stopWords = stopwords.words("english")
#print (stopWords)
exclude = set(string.punctuation)
#print (string.punctuation)
#print (exclude)
exclude.remove("_")

vocabList = set(w.lower() for w in nltk.corpus.words.words())
#print (vocabList)

phrasesDict = dict()
for a in range(len(reviewContent)):								#Stores the score of the nouns
		for i in range(len(reviewContent[a])):
			line_words = reviewContent[a][i]          
			phrases = TextBlob(line_words).noun_phrases
             
			for p in phrases:
				if(len(p.split()) == 2):
					if(p not in phrasesDict):
						phrasesDict[p] = 1
					else:
						phrasesDict[p] += 1
             
#print (phrasesDict)

phrasesDict = OrderedDict(sorted(phrasesDict.items(), key=operator.itemgetter(1), reverse=True))
#print(phrasesDict)
newPhrases = dict()

for line_words, count in phrasesDict.items():
		#Preprocessing text
		line_words = ' '.join([apostropheList[word] if word in apostropheList else word for word in line_words.split()])
		line_words = ''.join(ch for ch in line_words if ch not in exclude)
		line_words = re.sub(r' [a-z][$]? ', ' ', line_words)
		line_words = [Word(word).lemmatize() for word in line_words.split() if(word not in stopwords.words("english") and not word.isdigit()) and len(word) > 2]
		line_words = ' '.join(line_words)
		if(len(line_words.strip(" ").split()) == 2):
			if(line_words in newPhrases):
				newPhrases[line_words] += count
			else:
				newPhrases[line_words] = count
#print (newPhrases)
newPhrases = OrderedDict(sorted(newPhrases.items(), key=operator.itemgetter(1), reverse=True))
#print (newPhrases)

nouns1 = []

for key, value in newPhrases.items():
		if value >= 3:
			nouns1.append(key)
#print(nouns1)


tb = Blobber(pos_tagger=NLTKTagger())
#print(tb)
f = open('modified.txt', 'w')   


for a in range(len(reviewContent)):
    f.write("[t]")
    text=reviewTitle[a]
    x=tb(text).tags
    #print (x)
    e=0
    while e<len(x):
        tagList=""
        temp=""
        wrt=x[e][0]
        #print(wrt)
        e=e+1
        count=e
        tp=0
        if(count<len(x) and (x[count-1][1]=="NN" or "JJ") and (x[count][1] == "NN" or "JJ")):
            tagList=x[count-1][0] + " " + x[count][0]
            temp = x[count][0]
            count =count +1
            #print (tagList)
        if tagList!="":
            #print (tagList)
            if tagList in nouns1:
                #print (tagList)
                tagList=tagList.replace(' ','')
                #print (tagList)
                f.write(tagList)
                tp=1
                e=count
        if tp==0:
            f.write(wrt)
        f.write(" ")
    f.write("\r\n")
    for i in range(len(reviewContent[a])):
        text=reviewContent[a][i]
        x= tb(text).tags
        tagList= []
        e=0
        f.write("##")
        while e<len(x):
            tagList= ""
            temp=""
            wrt=x[e][0]
            #print(wrt)
            e=e+1
            count=e
            tp=0
            if(count<len(x) and (x[count-1][1]=="NN" or "JJ") and (x[count][1] == "NN" or "JJ")):
                tagList=x[count-1][0] + " " + x[count][0]
                temp=x[count][0]
                count=count+1
            if tagList!="":
                #print(tagList)
                if tagList in nouns1:
                    tagList=tagList.replace(' ','')
                    #print (tagList)
                    f.write(tagList)
                    tp=1
                    e=count
            if tp==0:
                f.write(wrt)
            f.write(" ")
        f.write(".\r\n")
















##hac


maxHops=4

nounScores = dict()
adjDict =dict()
for a in range(len(reviewContent)):
    for i in range(len(reviewContent[a])):
        text = ' '.join([word for word in reviewContent[a][i].split() if word not in stopwords.words("english")])
        text = ''.join(ch for ch in text if ch not in exclude)
        
        text = nltk.word_tokenize(text)
        #print(text)
        x = nltk.pos_tag(text)
        #print(x)
        
        tagList= []
        for e in x:
            if(e[1]=="NN" or e[1]=="JJ"):
                tagList.append(e)
        #print(tagList)
        
        for e in tagList:
            if e[1]=="NN":
                if e[0] not in nounScores:
                    nounScores[e[0]]=0
        
        for l in range(len(tagList)):
            if("JJ" in tagList[l][1]):
                j = k = leftHop = rightHop = - 1
                
                for j in range (l+1,len(tagList)):
                    if(j== l + maxHops):
                        break
                    if("NN" in tagList[j][1]):
                        rightHop=(j-l)
                        break
               
                for k in range(l-1,-1,-1):
                    
                    if(j==l-maxHops):
                        break
                    if("NN" in tagList[k][1]):
                        leftHop = (l-k)
                        break
                        
                
                if(leftHop > 0 and rightHop > 0):
                    if (leftHop-rightHop) >= 0:
                        adjDict[tagList[l][0]] = tagList[j][0]
                        nounScores[tagList[j][0]]+=1
                    else:
                        adjDict[tagList[l][0]] = tagList[k][0]
                        nounScores[tagList[k][0]]+=1
                elif leftHop > 0 :
                    adjDict[tagList[l][0]] = tagList[k][0]
                    nounScores[tagList[k][0]]+=1
                elif rightHop> 0 :
                    adjDict[tagList[l][0]] = tagList[j][0]
                    nounScores[tagList[j][0]]+=1
#print(adjDict)
nounScores = OrderedDict(sorted(nounScores.items(), key=operator.itemgetter(1)))
#print(nounScores)
#print(nouns1)   
adjectList = list(adjDict.keys())
#print(adjDict)
#print(adjectList)
nouns = []
for key, value in nounScores.items():
    if value >= 3:
        nouns.append(key)
               
for a in range(len(reviewContent)):
    f.write("[t]"+reviewTitle[a])
    f.write("\r\n")
    
    for i in range(len(reviewContent[a])):
        text = reviewContent[a][i]
        x = tb(text).tags
        tagList = []
        e=0
        f.write("##")
                
        while e < len(x):
            tagList = []
            f.write(x[e][0])
            #print(x[e][0])
            e=e+1
            count=e
            
            if(count<len(x) and x[count-1][1] == "NN" and x[count][1] == "NN"):
                tagList.append(x[count-1][0])
                
                while(count < len(x) and x[count][1] == "NN"):
                    tagList.append(x[count][0])
                    count= count+1
            #print(tagList)
            if tagList != [] and len(tagList) == 2:
                if set(tagList) <= set(nouns): 
                    for t in range(1,len(tagList)):
                        f.write(tagList[t])
                    e = count
            f.write(" ")
        f.write(".\r\n")
       
##############  ngraams
        

                    
                        
reviewContent1 =[]
reviewTitle1 =[]    

with open("modfied.txt") as f:
    review=[]
    for line in f:
        if line[:3] == "[t]":
            if review:
                reviewContent1.append(review)
                review=[]
            reviewTitle1.append(line.split("[t]")[1].rstrip("\r\n"))
        else:
            if "##" in line:
                x=line.split("##")
                for i in range(1,len(x)):
                    review.append(x[i].rstrip("\r\n"))
            else:
                continue
    reviewContent1.append(review)

#print(nouns)


################adj scores
#print(adjectList)
adjScores = dict()
for i in adjectList:
    blob= TextBlob(i)
    if(blob.sentiment.polarity!=0):
        adjScores[i]=blob.sentiment.polarity
adjScores.update((x, 4 * y) for x, y in adjScores.items())
#print(adjScores)

####################mos
import NOUN
featureList =nouns
alpha=0.6

pos_review_index = dict()
neg_review_index = dict()
neut_review_index = dict()


global_noun_scores= dict()

global_noun_adj_count = dict()
for a in range(len(reviewContent)):
    review_noun_scores =dict()
    title_noun_scores= dict()
    
    review_noun_adj_count =dict()
    title_noun_adj_count=dict()
    
    for lineIndex in range(len(reviewContent[a])+1):
        if(lineIndex == len(reviewContent[a])):
            line_words = reviewTitle[a]
        else:
            line_words = reviewContent[a][lineIndex]
        
        line_words = ' '.join([apostropheList[word] if word in apostropheList else word for word in line_words.split()])
        line_words = ''.join(ch for ch in line_words if ch not in exclude)
        line_words = re.sub(r' [a-z][$]? ', ' ', line_words)
        line_words = [word for word in line_words.split() if(word not in stopwords.words("english") and not word.isdigit()) and len(word) > 2]
        #print(line_words)
        for wordIndex in range(len(line_words)):
            word = line_words[wordIndex]
            
            if word in adjScores:
                score=adjScores[word]
                
                
                if(wordIndex -2 >=0):
                    
                    phrase = line_words[wordIndex - 2] + " " + line_words[wordIndex - 1] + " " + line_words[wordIndex]
                    
                    if((TextBlob(phrase).sentiment.polarity * score) < 0):
                        score*=-1
                elif(wordIndex-1>=0):
                    
                    phrase = line_words[wordIndex - 1] + " " + line_words[wordIndex]
                    
                    if((TextBlob(phrase).sentiment.polarity * score) < 0):
                        score*=-1
                
                closest_noun = NOUN.find_closest_noun(wordIndex, line_words, featureList)
                if(closest_noun is None):
                    continue
                
                if(lineIndex == len(reviewContent[a])):
                    if(closest_noun in title_noun_scores):
                        title_noun_scores[closest_noun] += score
                    else:
                        title_noun_scores[closest_noun] = score
                    
                    
                    if(closest_noun in title_noun_adj_count):
                        title_noun_adj_count[closest_noun] += 1
                    else:
                        title_noun_adj_count[closest_noun] = 1
                else:
                    
                    if(closest_noun in review_noun_scores):
                        review_noun_scores[closest_noun] += score
                    else:
                        review_noun_scores[closest_noun] = score
                    
                    if(closest_noun in global_noun_scores):
                        global_noun_scores[closest_noun] += score
                    else:
                        global_noun_scores[closest_noun] = score
                    
                    
                    
                    if(closest_noun in review_noun_adj_count):
                        review_noun_adj_count[closest_noun] += 1
                    else:
                        review_noun_adj_count[closest_noun] = 1
                    
                    if(closest_noun in global_noun_adj_count):
                        global_noun_adj_count[closest_noun] += 1
                    else:
                        global_noun_adj_count[closest_noun] = 1
    print("#########################")
    print(review_noun_scores)
    objects=review_noun_scores.keys()
    performance = review_noun_scores.values()
    
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Scores')
    plt.title('Features')
 
    plt.show()
                    
    total_score = sum(review_noun_scores.values())
    total_adj = sum(review_noun_adj_count.values())
    if(total_adj == 0):
        review_score = 0
    else:
        review_score = total_score / float(total_adj)
             

#print(global_noun_scores) 
    total_score = sum(review_noun_scores.values())
    total_adj = sum(review_noun_adj_count.values())
    
    if(total_adj == 0):
        review_score = 0
    else:
        review_score = total_score / float(total_adj)
    
    
    title_total_score = sum(title_noun_scores.values())
    title_total_adj = sum(title_noun_adj_count.values())
    if(title_total_adj == 0):
        title_score = 0
    else:
        title_score = title_total_score / float(title_total_adj)
    
    
    avg_score = ((alpha * title_score) + review_score) / (alpha + 1)
    
    
    
avg_feature_score = dict()
    
for noun in global_noun_scores:
    avg_feature_score[noun] = global_noun_scores[noun] / float(global_noun_adj_count[noun])
    
avg_feature_score = sorted(avg_feature_score.items(), key=operator.itemgetter(1), reverse=True)
    

objects=[]
performance = []
for a in range(len(avg_feature_score)):
    objects.append(avg_feature_score[a][0])

for a in range(len(avg_feature_score)):
    performance.append(avg_feature_score[a][1])

#print (performance)

#print(avg_feature_score)

    
   
    

          
##########################
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 

y_pos = np.arange(len(objects))

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Features')
 
plt.show()


    
