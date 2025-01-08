from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata
import streamlit as st
from matplotlib.patches import Polygon
import sys
from matplotlib.ticker import FixedLocator
from mplsoccer import PyPizza, FontManager, Radar
import warnings
import sys
if 'warnings' not in sys.modules:
    import warnings


    
#radar chart creation
def radar(player, position, stats, title):
    # Define the parameters and their ranges
    params = stats  # List of stat names
    low = [0] * len(params)  # Minimum value for each stat
    high = [100] * len(params)  # Maximum value for each stat (percentile range)

    # Player and Position Values
    player_values = [player[stat] * 100 for stat in stats]  # Player percentiles
    #position_values = [position[stat] * 100 for stat in stats]  # Positional average percentiles

    #create a Radar object
    radar = Radar(
        params=params,  # Parameter names
        min_range=low,  # Minimum value for each parameter
        max_range=high,  # Maximum value for each parameter
        num_rings=4,  # Number of concentric circles
        ring_width=1,  # Width of each ring
        center_circle_radius=1  # Radius of the center circle
    )

    #create the figure and axis for the radar chart
    fig, ax = radar.setup_axis()

    #plot the player's data
    radar_output = radar.draw_radar(
        player_values,
        ax=ax,
        kwargs_radar={'facecolor': 'blue', 'alpha': 0.5, 'edgecolor': 'blue', 'linewidth': 2}, #radar style
    )
    #add outlines
    max_radius = radar.ring_width * radar.num_rings  # Calculate the maximum radius
    for i in range(1, radar.num_rings + 2):
        radius = i  # Normalize radius based on number of rings
        circle = plt.Circle(
            (0, 0), radius,  # Set the center and radius
            transform=ax.transData._b,  # Use the correct transformation for the radar
            color="gray", linestyle="--", linewidth=1, fill=False
        )
        ax.add_artist(circle)
    #draw parameter (outer) labels
    radar.draw_param_labels(ax=ax, fontsize=14, color="black")

    # Draw range (inner) labels
    #radar.draw_range_labels(ax=ax, fontsize=10, color="gray")

    # Add title
    fig.text(
        0.5, 0.98, title,
        ha="center", fontsize=18, color="black", weight="bold"
    )

    return fig
    
    
#pizza chart:
def pizza(player,player_2,stats,title, name1=None,name2=None):
    num = len(stats)
    angles=np.linspace(0,2*np.pi,num,endpoint=False)
    angles_mids=angles+(angles[1]/2)
    player_vals = [round(player[stat] *100) for stat in stats]
    if player_2:
        player_2_vals = [round(player_2[stat] *100) for stat in stats]
        #get params_offset (True if the absolute difference is <10)
        params_offset = [abs(p1 - p2) < 10 for p1, p2 in zip(player_vals, player_2_vals)]

        #create pie chart
        baker = PyPizza(
            params=stats,                  #list of params (stats)
            straight_line_color="#000000", #color for straight lines
            straight_line_lw=1,            #linewidth for straight lines
            last_circle_lw=1,              #linewidth of last circle
            other_circle_lw=1,             #linewidth for other circles
            other_circle_ls="-."           #linestyle for other circles
        )

        #plot pizza for player1 and 2
        fig, ax = baker.make_pizza(
            player_vals,                      #player 1 values
            compare_values=player_2_vals,    #player 2 values
            figsize=(10, 10),                   #adjust figsize
            param_location=110,               #where the parameter labels will be added
            kwargs_slices=dict(
                facecolor="yellow", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                                #Player 1 slice appearance
            kwargs_compare=dict(
                facecolor="blue", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                                #Player 2 slice appearance
            kwargs_params=dict(
                color="#000000", fontsize=16,
                va="center"
            ),                                #Param label appearance
            kwargs_values=dict(
                color="#000000", fontsize=14, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="yellow",
                    boxstyle="round,pad=0.2", lw=1
                )
            ),                                #Player 1 values appearance
            kwargs_compare_values=dict(
                color="#000000", fontsize=14, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="cornflowerblue",
                    boxstyle="round,pad=0.2", lw=1
                )
            )                                 #Player 2 values appearance
        )

        #close value text adjustment from mplsoccer
        baker.adjust_texts(params_offset, offset=-0.2, adj_comp_values=True)
        #add legend
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color="yellow", lw=4, label=f"{name1}"),
                plt.Line2D([0], [0], color="lightblue", lw=4, label=f"{name2}")
            ],
            loc="lower left",
            bbox_to_anchor=(-0.2, -0.1),
            fontsize=14,
            frameon=False
        )
    else:
        #raw_values = [raw_player_values[stat] for stat in stats]
        #create plot
        baker = PyPizza(
            params=stats,                  
            straight_line_color="#000000", 
            straight_line_lw=1,            
            last_circle_lw=1,              
            other_circle_lw=1,             
            other_circle_ls="-."           
        )

        # Plot pizza for player
        fig, ax = baker.make_pizza(
            player_vals,                 #player values
            figsize=(8, 8),                
            param_location=110,            
            kwargs_slices=dict(
                facecolor="yellow", edgecolor="#000000",
                zorder=2, linewidth=1
            ),                             #slices appearance
            kwargs_params=dict(
                color="#000000", fontsize=12,
                va="center"
            ),                             #param label appearance
            kwargs_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor="yellow",
                    boxstyle="round,pad=0.2", lw=1
                )
            )                              
        )

    #add title
    fig.text(
        0.515, 0.95, title, size=24,
        ha="center", color="#000000"
    )

    #add credits
    CREDIT_1 = "Program Written By Adith George; Data Taken From FBRef (StatsBomb)"
    CREDIT_2 = "Inspired by mplsoccer visualizations and work by Naveen Elliott & Liam Henshaw"

    fig.text(
        0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
        color="#000000", ha="right"
    )

    return fig

#similarity calc
def similarity(df,name,position,stats,threshold=5):
    player=df[df["Player"]==name]
    if player.empty: #null case
        st.error(f"{name} not found")
        return None
    
    #ensure player is present in position df
    if player["Position"].iloc[0] != position:
        player["Position"] = position
        df = pd.concat([df,player], ignore_index=True)
    position_data = df[df["Position"] == position].copy()

    #calculate percentiles
    for stat in stats:
        position_data[f"{stat}_Percentile"] = rankdata(position_data[stat], method="average") / len(position_data)
        #account for statistics where lower numbers = better performance like errors and GA
        inverse_stats = ["Errors Per 90", "Goals Allowed Per 90", "GA/SoT Per 90"]
        if stat in inverse_stats:
            position_data[f"{stat}_Percentile"] = 1 - position_data[f"{stat}_Percentile"]
    #extract the percentiles for the selected player
    player_percentiles = position_data[position_data["Player"] == name][[f"{stat}_Percentile" for stat in stats]]

    if player_percentiles.empty:
        st.error(f"No percentile data found for player '{name}' in position '{position}'.")
        return None

    player_percentiles = player_percentiles.iloc[0].to_numpy().reshape(1, -1)

    # Use percentiles for similarity calculation
    similarity_scores = cosine_similarity(
        position_data[[f"{stat}_Percentile" for stat in stats]], player_percentiles
    ).flatten()
    position_data["Similarity"] = similarity_scores

    # Sort by similarity
    position_data = position_data.sort_values("Similarity", ascending=False)

    # Add raw values for radar stats
    radar_stats = stats  # Use the raw stats for display purposes
    output_columns = ["Player", "Team", "Age","Position", "Similarity"] + radar_stats

    return position_data[output_columns].head(threshold)


def load():
    import warnings
    if 'warnings' not in sys.modules:
        import warnings
    return pd.read_csv("mls_players.csv")
def load_gk():
    import warnings
    if 'warnings' not in sys.modules:
        import warnings
    return pd.read_csv("mls_gk.csv")
    
#create app
def main():
    st.title("MLS Player Comparer")
    f = load()
    g=load_gk()
    pl = f.copy()
    gk=g.copy()
    if pl.empty or gk.empty:
        st.error("Dataset could not be loaded. Please check the file path.")
        return
    st.sidebar.header("Player Search and Filters")
    name = st.sidebar.text_input("Enter a Player's Name")
    compare = st.sidebar.text_input("Enter a Player to Compare To (Optional)")
    position = st.sidebar.selectbox("Select Position",options=["FW","MF","DF","GK"])
    chart_type = st.sidebar.selectbox("Select Chart Type",options=["Radar", "Pizza"])
    threshold = st.sidebar.slider("# Similar Players", 0, 10,1)
    st.sidebar.info(
        """
        **Note**: Currently only have comparison pizza graphs, not radar.
        Also, percentiles for certain stats *Errors Per 90*, *Goals Allowed Per 90*, 
        and *GA/SoT Per 90* are **inverted**. This means lower raw values for these stats 
        correspond to higher percentiles, as lower values indicate better performance.
        """
    )
    df = gk if position=="GK" else pl
    if name:
        player=df[df["Player"]==name]
        if not player.empty:
            if position == "FW":
                stats = ["Goals per 90", "xG per 90", "Assists Per 90",  "xAG per 90", 
                         "Aerial Win %", "Shot Creating Actions per 90", "SoT per 90", "Take-Ons Attempted per 90", "Passes Completed Per 90"]
            elif position == "MF":
                stats = ["Carries per 90", "Successful Take-On %", "Pass Completion Rate", "Tackles Won %", 
                         "Shot Creating Actions per 90", "Dribble Distance per 90", "Goals per Shot", "Interceptions Per 90", "Total Prog Pass Distance per 90", "Prog Passes Received Per 90"]
            elif position == "DF":
                stats = ["Tackles Won %", "Shot Creating Actions per 90", "npXG+xAG per 90", "Progressive Passes Per 90", 
                         "Errors Per 90", "Passes Completed Per 90", "Dribble Distance per 90", "Pass 15-30 Comp %", "Aerial Win %", "Long_Cmp%"]
            else:
                stats = ["Save %", "PSxG-GA per 90", "Pass Completion Rate","Errors Per 90", "Crosses Stopped %", "Avg Pass Length", "GA/SoT Per 90", "Goals Allowed Per 90", "Pass >40 Yards Cmp%"]
            position_data = df[df["Position"] == position].copy()
            #temporarily add the player to the position data if positions don't match
            if player["Position"].iloc[0] != position:
                if position == "GK":
                    st.warning(f"{name} is not a goalkeeper, except Sean Zawadzki :)")
                else:
                    temp_player = player.copy()
                    temp_player["Position"] = position
                    position_data = pd.concat([position_data, temp_player], ignore_index=True)
            #check if comparison player exists
            if compare:
                st.header(f"{name} vs. {compare}")
                player_2 = df[df["Player"]==compare]
                if player_2.empty:
                    st.warning(f"{compare} not found")
                #temporarily add the player to the position data if positions don't match
                if player_2["Position"].iloc[0] != position:
                    temp_player = player_2.copy()
                    temp_player["Position"] = position
                    position_data = pd.concat([position_data, temp_player], ignore_index=True)
            else:
                  st.header(f"{name}")
            for stat in stats:
                position_data[f"{stat}_Percentile"] = rankdata(position_data[stat], method="average") / len(position_data)
                #inverse/lower score=better stats
                inverse_stats = ["Errors Per 90", "Goals Allowed Per 90", "GA/SoT Per 90"]
                if stat in inverse_stats:
                    position_data[f"{stat}_Percentile"] = 1 - position_data[f"{stat}_Percentile"]
            player_percentiles = position_data[position_data["Player"] == name][[f"{stat}_Percentile" for stat in stats]].iloc[0].to_dict()
            if compare:
                player_2_percentiles = position_data[position_data["Player"] == compare][[f"{stat}_Percentile" for stat in stats]].iloc[0].to_dict()
            position_percentiles = position_data[[f"{stat}_Percentile" for stat in stats]].mean().to_dict()

            #adjust keys for radar chart input
            player_stats = {stat: player_percentiles[f"{stat}_Percentile"] for stat in stats}
            if compare:
                player_2_stats = {stat: player_2_percentiles[f"{stat}_Percentile"] for stat in stats}
            position_stats = {stat: position_percentiles[f"{stat}_Percentile"] for stat in stats}
            #create radar chart
            if stats:
                if compare:
                    title = f"{name} vs {compare} - {position} Comparison"
                    fig = pizza(player_stats, player_2_stats, stats, title, name, compare)
                    st.pyplot(fig)
                elif chart_type == "Pizza":
                    fig=pizza(player_stats,None,stats,f"{name} - {position} Analysis")
                    st.pyplot(fig)
                elif chart_type == "Radar":
                    fig=radar(player_stats,position_stats,stats,f"{name} - {position} Analysis")
                    st.pyplot(fig)
                else:
                    st.warning("Choose Chart Type")
            else:
                st.warning("No stats available for this position")
            #get similar players
            similar = similarity(df,name,position,stats, threshold)
            if similar is not None and not similar.empty:
                st.subheader("Similar Players")
                st.dataframe(similar)
            else:
                st.error("Player not found")
if __name__ == "__main__":
    main()
                
        