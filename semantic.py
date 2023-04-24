import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

#comparing words
print('-------------Comparing words----------------')
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

print('\n')
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


print('\n')
#comparing sentences
print('-------------Comparing sentences----------------')
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#Questions

#What you found interesting about the similarities between cat, monkey and banana
"""What I found interesting about the similarities between the cat, monkey and banana is that they can be associated with each other based on their
attributes or characteristics, such as type of organisms,  their role in the environment etc, behaviour etc. For example, the cat and the monkey have a 
high similarity as they are the same type of organisms, they are both animals. Monkey and banana similarity is based on the fact that banana's are a monkey's favourite food.
Another example can be a cat and dog, they would have a higher similarity because they are both house pets"""

#Run the example file with the simpler language model ‘en_core_web_sm’ and write a note on what you notice is different from the model 'en_core_web_md'.
"""The results are different when the ‘en_core_web_sm’ model is used. The results are less accurate because the ‘en_core_web_sm’ is less accurate in breaking down complex sentences as it has
poor vocabulary. The similarities are based entity comparisons, which provides less accurate results"""