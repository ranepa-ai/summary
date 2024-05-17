from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import requests
import Levenshtein
import pandas as pd
from flask import Flask, request, jsonify

class PromptMaster:
    def __init__(self, api_url="http://localhost:5001/generate"):
        self.api_url = api_url
        with open("prompt_list.txt", "r", encoding="utf-8") as file:
            self.prompt_sum = file.read()
        self.prompt_perefrase = "Перефразируй данный текст, чтобы получилось 1 предложение, донеси основную мысль о тексте и действующие события: {}\nРезультат: "
        self.cos_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    def get_request(self, text):
        payload = {"prompt": text}
        response = requests.post(self.api_url, json=payload)
        generated_text = response.text
        return generated_text

    def gen_summary(self, text):
        prompt = self.prompt_sum.format(text[:2000])
        summary = self.get_request(prompt)
        #result = self.prompt_perefrase.format(summary)
        #result = self.get_request(result)
        return summary
    
    def get_metrics(self, sentences):
        embeddings = self.cos_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        cosine_distance = similarity_matrix[0, 1]
        fuzz_metric = fuzz.WRatio(sentences[0], sentences[1])
        lev_metric = Levenshtein.distance(sentences[0], sentences[1])
        return [cosine_distance, fuzz_metric, lev_metric]

app = Flask(__name__)
PM = PromptMaster()

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        text_to_sum = data['text_to_sum']
        predict_summary = PM.gen_summary(text_to_sum[:1000])
        return jsonify({'predict_summary': predict_summary})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='83.143.66.61', port=27370)
