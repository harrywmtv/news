import feedparser
from transformers import pipeline
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# List of supported countries with their codes
COUNTRIES = {
    'United States': {'gl': 'US', 'hl': 'en-US', 'ceid': 'US:en'},
    'United Kingdom': {'gl': 'GB', 'hl': 'en-GB', 'ceid': 'GB:en'},
    'Canada': {'gl': 'CA', 'hl': 'en-CA', 'ceid': 'CA:en'},
    'Australia': {'gl': 'AU', 'hl': 'en-AU', 'ceid': 'AU:en'},
    'India': {'gl': 'IN', 'hl': 'en-IN', 'ceid': 'IN:en'},
    'Germany': {'gl': 'DE', 'hl': 'de', 'ceid': 'DE:de'},
    'France': {'gl': 'FR', 'hl': 'fr', 'ceid': 'FR:fr'},
    'Japan': {'gl': 'JP', 'hl': 'ja', 'ceid': 'JP:ja'},
    'China': {'gl': 'CN', 'hl': 'zh-CN', 'ceid': 'CN:zh-Hans'},
    'Brazil': {'gl': 'BR', 'hl': 'pt-BR', 'ceid': 'BR:pt-419'}
    # Add more countries as needed
}

# Initialize the sentiment analysis pipeline with FinBERT globally
classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')

def get_top_headlines_with_sentiment(country_code):
    # Get the country parameters
    country_params = COUNTRIES.get(country_code, COUNTRIES['United States'])
    gl = country_params['gl']
    hl = country_params['hl']
    ceid = country_params['ceid']

    # URL of the Google News RSS feed for the selected country
    rss_url = f'https://news.google.com/rss?hl={hl}&gl={gl}&ceid={ceid}'

    # Parse the RSS feed
    feed = feedparser.parse(rss_url)

    # Check if the feed was successfully parsed
    if feed.bozo:
        print("Failed to parse RSS feed.")
        return []

    headlines_with_sentiment = []

    # Get the top 10 entries (headlines)
    if feed.entries:
        for idx, entry in enumerate(feed.entries[:10], start=1):
            headline = entry.title
            # Perform sentiment analysis
            result = classifier(headline)[0]
            sentiment = result['label']
            confidence = result['score']
            # Map sentiment labels to Positive/Negative/Neutral
            if sentiment == 'positive':
                market_sentiment = 'Positive'
            elif sentiment == 'negative':
                market_sentiment = 'Negative'
            else:
                market_sentiment = 'Neutral'

            headlines_with_sentiment.append({
                'index': idx,
                'headline': headline,
                'sentiment': market_sentiment,
                'confidence': confidence
            })
    else:
        print("No news entries found.")
    return headlines_with_sentiment

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_country = request.args.get('country', 'United States')
    headlines = get_top_headlines_with_sentiment(selected_country)
    current_year = datetime.now().year
    return render_template(
        'index.html',
        headlines=headlines,
        current_year=current_year,
        countries=COUNTRIES,
        selected_country=selected_country
    )

if __name__ == "__main__":
    app.run(debug=True)
