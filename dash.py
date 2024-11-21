import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from wordcloud import WordCloud
import re
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
# Load the saved model and vectorizer
model = load_model('sentimental analysis model.h5')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Function to make sentiment prediction
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    input_vector = vectorizer.transform([processed_text])
    prediction = model.predict(input_vector.toarray())
    return "Positive" if prediction >= 0.5 else "Negative"

# Load and preprocess data
df_train = pd.read_csv('train.tsv', sep='\t', header=None)
df_train.columns = ['sentiment', 'analysis']

df_test = pd.read_csv('test.tsv', sep='\t', header=None)
df_test.columns = ['sentiment', 'analysis']

df_dev = pd.read_csv('dev.tsv', sep='\t', header=None)
df_dev.columns = ['sentiment', 'analysis']

data = pd.concat([df_train, df_test, df_dev])

def generate_wordcloud_image(df):
    positive_reviews = df[df['analysis'] == 1]['sentiment']  # Ensure correct filtering
    if positive_reviews.empty:
        raise ValueError("No positive reviews found in the dataset.")
    
    positive_text = " ".join(positive_reviews)  # Join the reviews into one string
    all_words = ' '.join(positive_text).lower()
    all_words = re.findall(r'\b\w+\b', all_words)  # Extract valid words
    
    if not all_words:
        raise ValueError("No valid words found in positive reviews.")
    
    wordcloud = WordCloud(background_color='white', max_words=50).generate(positive_text)

    # Convert to PNG
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Generate the word cloud for positive reviews
try:
    wordcloud_image = generate_wordcloud_image(data)
except ValueError as e:
    print(f"Error generating word cloud: {e}")

# Function to generate the Seaborn countplot as an image
def generate_class_countplot_image(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='analysis', data=df, palette="viridis")
    plt.title("Class Distribution (Positive vs Negative)")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# Function to generate the pie chart as an image
def generate_pie_chart_image(df):
    class_counts = df['analysis'].value_counts()
    
    plt.figure(figsize=(6, 6))
    myexplode = [0.2, 0]  # Add an explode effect to highlight the first slice
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, explode=myexplode, shadow=True)
    plt.title("Class Distribution (Positive vs Negative)")
    plt.axis('equal')  # Ensures pie is drawn as a circle
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# Generate plots as images
countplot_image = generate_class_countplot_image(data)
pie_chart_image = generate_pie_chart_image(data)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Sentiment Analysis Dashboard", className='text-center text-white mb-4'), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Input(id='text-input', type='text', placeholder='Enter text here...', style={'fontSize': 24, 'width': '80%'}),
            html.Button('Analyze', id='analyze-button', n_clicks=0, className='btn btn-primary', style={'fontSize': 24, 'marginTop': 10}),
            html.Div(id='predicted-sentiment', className='mt-3', style={'fontSize': 20, 'textAlign': 'center'})
        ], width=12, style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexDirection': 'column'})
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            html.H3("Class Distribution - Count Plot", className="text-light"),
            html.Img(src=f'data:image/png;base64,{countplot_image}', style={'width': '80%'}),
        ], width=6, className="text-center"),
        dbc.Col([
            html.H3("Class Distribution - Pie Chart", className="text-light"),
            html.Img(src=f'data:image/png;base64,{pie_chart_image}', style={'width': '80%'}),
        ], width=6, className="text-center")
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.H3("Most Common Words in Positive Reviews", className="text-light"),
            html.Img(src=f'data:image/png;base64,{wordcloud_image}', style={'width': '60%'}),  # Adjusted size for the wordcloud
        ], width=12, className="text-center")
    ], className="mt-4")
], fluid=True)

# Callback to handle prediction
@app.callback(
    Output('predicted-sentiment', 'children'),
    Input('analyze-button', 'n_clicks'),
    Input('text-input', 'value')
)
def update_prediction(n_clicks, text):
    if n_clicks > 0 and text:
        sentiment = predict_sentiment(text)
        return f"Predicted sentiment: {sentiment}"
    return "Enter text and click Analyze."

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, dev_tools_props_check=False)