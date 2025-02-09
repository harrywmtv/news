import feedparser
from transformers import pipeline
from flask import Flask, render_template, request, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from datetime import datetime
import logging
import os

app = Flask(__name__)
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Cache for feed entries
feed_cache = {}

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

def validate_country(country_code):
    """Ensure the country code is valid."""
    return country_code if country_code in COUNTRIES else 'United States'

def fetch_feed_entries(country_code):
    """Fetch and parse RSS feed entries for the given country."""
    country_params = COUNTRIES[country_code]
    gl, hl, ceid = country_params['gl'], country_params['hl'], country_params['ceid']
    rss_url = f'https://news.google.com/rss?hl={hl}&gl={gl}&ceid={ceid}'
    feed = feedparser.parse(rss_url)
    if not feed.bozo and feed.entries:
        return feed.entries
    else:
        logging.error("Failed to parse RSS feed.")
        return []

def get_top_headlines_with_sentiment(country_code, page=1, page_size=30):
    country_code = validate_country(country_code)
    cache_key = country_code

    if cache_key not in feed_cache:
        feed_cache[cache_key] = fetch_feed_entries(country_code)

    entries = feed_cache.get(cache_key, [])
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    if start_idx >= len(entries):
        return []

    headlines_with_sentiment = []
    
    for idx, entry in enumerate(entries[start_idx:end_idx], start=start_idx + 1):
        try:
            result = classifier(entry.title)[0]
            sentiment = result['label']
            confidence = result['score']
            market_sentiment = {
                'positive': 'Positive',
                'negative': 'Negative'
            }.get(sentiment, 'Neutral')

            headlines_with_sentiment.append({
                'index': idx,
                'headline': entry.title,
                'sentiment': market_sentiment,
                'confidence': confidence
            })
        except Exception as e:
            logging.error(f"Sentiment analysis failed for headline: {entry.title}. Error: {e}")

    return headlines_with_sentiment

@app.route('/', methods=['GET'])
def index():
    selected_country = validate_country(request.args.get('country', 'United States'))
    headlines = get_top_headlines_with_sentiment(selected_country, page=1)
    current_year = datetime.now().year
    return render_template(
        'index.html',
        headlines=headlines,
        current_year=current_year,
        countries=COUNTRIES,
        selected_country=selected_country
    )

@app.route('/load_more', methods=['GET'])
def load_more():
    page = int(request.args.get('page', 1))
    selected_country = validate_country(request.args.get('country', 'United States'))
    headlines = get_top_headlines_with_sentiment(selected_country, page=page)
    return render_template('partials/headlines.html', headlines=headlines)

if __name__ == "__main__":
    app.run(host='0.0.0.0')  # Removed debug=True

# For production with Gunicorn
application = DispatcherMiddleware(
    Flask('dummy_app'),  # Dummy app for root
    {
        '/news': app
    }
)