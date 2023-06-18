from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
classifier = pipeline("sentiment-analysis")
set_seed(42)
numOfResults=5
results=generator("Hello, I'm a language model,", max_length=20, num_return_sequences=numOfResults)
for k in range(numOfResults):
    print(str(results[k]['generated_text'])+ str(classifier(results[k]['generated_text'])))