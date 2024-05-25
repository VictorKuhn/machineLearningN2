import pandas as pd
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')


# Função para carregar o dataset de filmes
def load_movies_dataset(file_path):
    movies_df = pd.read_csv(file_path)
    movies_df = movies_df[['title', 'description']].dropna()
    movies_df['description'] = movies_df['description'].astype(str)
    return movies_df


# Função para criar um dataset fictício de usuários sem repetição de gênero e sentimento
def create_users_dataset():
    user_ids = [f"user_{i}" for i in range(1, 6)]
    names = [f"User_{i}" for i in range(1, 6)]
    ages = [random.randint(18, 65) for _ in range(5)]
    genres = ["Action", "Comedy", "Drama", "Horror", "Romance"]
    preferred_sentiments = ["Anxiety", "Fear", "Excitement", "Thrill", "Attachment", "Motivation", "Love"]

    random.shuffle(genres)
    random.shuffle(preferred_sentiments)

    users = []

    for i in range(5):
        genre = genres[i]
        sentiment = preferred_sentiments[i]
        users.append({
            'user_id': user_ids[i],
            'name': names[i],
            'age': ages[i],
            'preferred_genre': genre,
            'preferred_sentiment': sentiment
        })

    users_df = pd.DataFrame(users)
    return users_df


# Função para gerar sentimentos das descrições dos filmes
def generate_sentiments(movies_df):
    sid = SentimentIntensityAnalyzer()
    sentiments = []

    for description in movies_df['description']:
        sentiment_score = sid.polarity_scores(description)
        sentiments.append(sentiment_score)

    sentiment_df = pd.DataFrame(sentiments)
    movies_df = pd.concat([movies_df, sentiment_df], axis=1)
    return movies_df


# Função para mapear os sentimentos de VADER para os sentimentos descritos
def map_sentiments(description, preferred_sentiment, sid):
    if isinstance(description, float):
        description = str(description)
    scores = sid.polarity_scores(description)

    sentiment_mapping = {
        'Anxiety': 'neg',
        'Fear': 'neg',
        'Excitement': 'pos',
        'Thrill': 'pos',
        'Attachment': 'pos',
        'Motivation': 'pos',
        'Love': 'pos'
    }

    return scores[sentiment_mapping[preferred_sentiment]]


# Função para recomendar filmes com base nas preferências do usuário
def recommend_movies(user_id, users_df, movies_df, sid):
    user_data = users_df[users_df['user_id'] == user_id].iloc[0]
    preferred_sentiment = user_data['preferred_sentiment']

    movies_df['similarity'] = movies_df['description'].apply(
        lambda desc: map_sentiments(desc, preferred_sentiment, sid))
    recommended_movies = movies_df.sort_values(by='similarity', ascending=False).head(10)

    return recommended_movies[['title', 'description', 'similarity']]


file_path = 'dataset_movies.csv'

# Carregar o dataset de filmes
movies_df = load_movies_dataset(file_path)

# Criar o dataset de usuários sem repetição de gênero e sentimento
users_df = create_users_dataset()

# Gerar sentimentos das descrições dos filmes
movies_df = generate_sentiments(movies_df)

# Criar o analisador de sentimentos
sid = SentimentIntensityAnalyzer()

# Gerar recomendações para os 5 usuários e salvar em um único CSV
all_recommendations = pd.DataFrame()

for i in range(1, 6):
    user_id = f'user_{i}'
    recommended_movies = recommend_movies(user_id, users_df, movies_df, sid)
    recommended_movies['user_id'] = user_id
    all_recommendations = pd.concat([all_recommendations, recommended_movies])

print(all_recommendations)

all_recommendations.to_csv('all_recommendations.csv', index=False)

movies_df.to_csv('movies_dataset.csv', index=False)
users_df.to_csv('users_dataset.csv', index=False)
