import requests

# URL of the Flask server
BASE_URL = 'http://127.0.0.1:5000'

# Function to get the list of players sorted by predicted future production
def get_future_production(position=None):
    url = f'{BASE_URL}/predict-future-production'
    params = {}
    if position:
        params['position'] = position
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# Function to draft a player (removes the player from the dictionary)
def draft_player(players_dict, player_name):
    if player_name in players_dict:
        del players_dict[player_name]
        return True
    else:
        return False

# Function to run the draft
def run_draft():
    players_list = get_future_production()
    
    if not players_list:
        print("No players available to draft.")
        return

    players_dict = {player['name']: player for player in players_list}
    
    while players_dict:
        # Display top players
        print("\nTop available players:")
        top_players = list(players_dict.values())[:10]
        for idx, player in enumerate(top_players, start=1):
            print(f"{idx}. {player['name']} - {player['team']} - {player['position']} - Predicted FPPG: {player['predicted_2024']}")
        draft_choice = input("Enter the name of the player to draft (or type 'exit' to stop): ").strip()
        
        if draft_choice.lower() == 'exit':
            break
        
        if draft_player(players_dict, draft_choice):
            print(f"{draft_choice} has been drafted.")
        else:
            print(f"Player {draft_choice} is not available. Please choose a player from the list.")

        if not players_dict:
            print("All players have been drafted!")

if __name__ == '__main__':
    run_draft()
