from flask import Flask, jsonify, request
import pandas as pd
import os
import numpy as np

app = Flask(__name__)

# Load and process the CSV files
def load_data():
    # Base directory for data files
    base_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # List of CSV files to load
    csv_files = [
        os.path.join(base_dir, '21-22skaters.csv'),
        os.path.join(base_dir, '22-23skaters.csv'),
        os.path.join(base_dir, '23-24skaters.csv')
    ]
    
    # Load each CSV file into a DataFrame
    dfs = [pd.read_csv(file) for file in csv_files]
    
    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df

# Function to calculate past points
def calculate_fantasy_points(df):
    ppp_df = df[df['situation'] == '5on4'].groupby('playerId').agg({
        'I_F_goals': 'sum',
        'I_F_primaryAssists': 'sum',
        'I_F_secondaryAssists': 'sum'
    }).reset_index()

    pk_df = df[df['situation'] == '4on5'].groupby('playerId').agg({
        'I_F_goals': 'sum',
        'I_F_primaryAssists': 'sum',
        'I_F_secondaryAssists': 'sum'
    }).reset_index()
    
    ppp_df['PPP'] = ppp_df['I_F_goals'] + ppp_df['I_F_primaryAssists'] + ppp_df['I_F_secondaryAssists']
    pk_df['PKP'] = pk_df['I_F_goals'] + pk_df['I_F_primaryAssists'] + pk_df['I_F_secondaryAssists']

    # Group by player and season
    df = df.groupby(['playerId', 'season']).agg({
        'I_F_goals': 'sum',
        'I_F_primaryAssists': 'sum',
        'I_F_secondaryAssists': 'sum',
        'I_F_shotsOnGoal': 'sum',
        'I_F_hits': 'sum',
        'shotsBlockedByPlayer': 'sum',
        'games_played': 'mean',
        'name': 'first',
        'team': 'last',
        'position': 'first'
    }).reset_index()
    
    df = pd.merge(df, ppp_df[['playerId', 'PPP']], on='playerId', how='left').fillna(0)
    df = pd.merge(df, pk_df[['playerId', 'PKP']], on='playerId', how='left').fillna(0)
    df['total_assists'] = df['I_F_primaryAssists'] + df['I_F_secondaryAssists']
    
    df['fantasy_points'] = (
        df['I_F_goals'] * 6 +
        df['total_assists'] * 4 +
        df['PPP'] * 1 +
        df['PKP'] * 2 +
        df['I_F_shotsOnGoal'] * 0.6 +
        df['I_F_hits'] * 0.2 + 
        df['shotsBlockedByPlayer'] * 0.5
    )

    df['fantasy_ppg'] = df['fantasy_points'] / df['games_played']

    
    return df

def predict_future_production(df):
    df = calculate_fantasy_points(df)
    grouped = df.groupby(['playerId', 'season']).agg({
        'fantasy_ppg': 'sum',
        'games_played': 'sum'
    }).reset_index()
    
    # Filter players that don't play enough to judge
    grouped = grouped[grouped['games_played'] >= 60]
    pivot_df = grouped.pivot(index='playerId', columns='season', values='fantasy_ppg')
    
    # Weight more recent seasons more heavily
    pivot_df['weighted_trend'] = (pivot_df[2023] - pivot_df[2022]) * 0.7 + (pivot_df[2023] - pivot_df[2021]) * 0.3
    
    # Calculate the trend, handling cases where the previous season's value is zero or very low
    pivot_df['trend'] = pivot_df['weighted_trend'] / (pivot_df[2023] + 1)
    
    # Apply scaling based on the player's original performance level
    pivot_df['scaling_factor'] = np.where(
        pivot_df[2023] > 0,
        1 / (1 + np.log1p(pivot_df[2023] / 10)),
        0
    )
    pivot_df['scaled_trend'] = pivot_df['trend'] * pivot_df['scaling_factor']
    
    # Apply a cap to the scaling factor to prevent unrealistic jumps for low-performing players
    pivot_df['scaled_trend'] = np.clip(pivot_df['scaled_trend'], -0.3, 0.3)
    
    # Predict future production (2024) by applying the scaled trend to the 2023 production
    pivot_df['predicted_2024'] = pivot_df[2023] * (1 + pivot_df['scaled_trend'])
    pivot_df['predicted_2024'] = np.maximum(pivot_df['predicted_2024'], pivot_df[2023] * 0.95)
    
    # Reduce the momentum of rebounds
    pivot_df['predicted_2024'] = np.where(
        (pivot_df[2022] > pivot_df[2021]) & (pivot_df[2023] > pivot_df[2022]),
        pivot_df['predicted_2024'] * 0.9,
        pivot_df['predicted_2024']
    )
    
    result_df = pivot_df[['predicted_2024']].merge(df[['playerId', 'name', 'team', 'position']].drop_duplicates(), on='playerId')
    
    result_df = result_df.groupby(['playerId', 'name']).agg({
        'predicted_2024': 'max',
        'position': 'first',
        'team' : 'last'
    }).reset_index()
    
    return result_df.sort_values(by='predicted_2024', ascending=False)


player_stats_df = load_data()

@app.route('/')
def home():
    return "Welcome to the Fantasy Hockey Draft Helper!"

@app.route('/calculate-fantasy-points', methods=['GET'])
def get_past_production():
    result_df = calculate_fantasy_points(player_stats_df)
    
    # Filter by position
    position = request.args.get('position')
    if position:
        result_df = result_df[result_df['position'] == position]
    
    top_players = result_df.sort_values(by='fantasy_ppg', ascending=False).head(100)
    top_players_dict = top_players[['name', 'team', 'position', 'season','fantasy_ppg']].to_dict(orient='records')
    
    return jsonify(top_players_dict)

@app.route('/predict-future-production', methods=['GET'])
def get_future_production():
    result_df = predict_future_production(player_stats_df)
    
    # Filter by position
    position = request.args.get('position')
    if position:
        result_df = result_df[result_df['position'] == position]
    
    top_players_dict = result_df[['name', 'team', 'position', 'predicted_2024']].head(100).to_dict(orient='records')
    
    return jsonify(top_players_dict)

if __name__ == '__main__':
    app.run(debug=True)
