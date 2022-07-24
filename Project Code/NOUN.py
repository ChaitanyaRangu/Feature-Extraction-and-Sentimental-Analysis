"""
  The MOS algorithm implementation - Classification of reviews and calculation of feature scores
"""

#Find the closest feature for an adj. Assumes a noun is found within 3 steps from the adj.
def find_closest_noun(wordIndex, line_words, features):
	ptr = 1
	while(ptr <= 3):
		if(wordIndex + ptr < len(line_words) and line_words[wordIndex + ptr] in features):
			return line_words[wordIndex + ptr]
		elif(wordIndex - ptr >= 0 and line_words[wordIndex - ptr] in features):
			return line_words[wordIndex - ptr]
		else:
			ptr += 1

