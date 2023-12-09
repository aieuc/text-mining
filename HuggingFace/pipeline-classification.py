from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I hate you")[0]
print(result)
result = classifier("I love you")[0]
print(result)

