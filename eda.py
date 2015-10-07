import pandas as pd
import nfldb_tables

if __name__ == '__main__':
    games, pp, player = nfldb_tables.get(['game', 'play_player', 'players'])
