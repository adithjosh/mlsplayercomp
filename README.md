# mlsplayercomp
URL: https://mls2324comp.streamlit.app/
Python App that creates radar or pizza charts for MLS Players with 4 or more starts in the 2024 MLS season. Allows for comparison and similarity table creation.

This streamlit app uses a dataset I created based on MLS 23/24 data from FBRef and utilizes the mplsoccer package to create Pizza and Python charts of MLS players alongside tables for similar players and best players per position based on chosen stats.

The user can provide a player (or two for comparison pizza chart) and the position they want to analyze the player's statistics with respect to other players in the position via percentiles. They can also select to use either a radar or pizza chart if only looking at one player, but for comparing two players a pizza chart will be selected by default as that is currently the only option for comparison. The user can also select the number of players they want to compare with the chosen player via a slider which will then create a table displaying the chosen amount of the most similar players and their respective critical stats based on the position chosen. There are also nation and league filters that restrict the set of players to start from before performing the percentile calculations. Finally, the user can choose a position and set of stats (along with the league and nation filter from the similar players) to view the best MLS players according to those stats using average percentile.

You can now select whatever stats you want to compare for the chosen position with the Stats dropdown in the sidebar.

It should be noted that inverse percentiles for errors and goals allowed related stats are used as for those stats, lower values indicate better performance overall.

Future plans include adding the ability for the user to choose the statistics they want in the radar chart (complete), adding filters for nation and team, teamwide comparisons, and adding data from more leagues (this may end up being a separate app).

For forwards, the default critical stats are Goals per 90, Expected Goals (xG) per 90, Assists Per 90, Expected Assisted Goals (xAG) per 90, Aerial Dual Win %, Shot Creating Actions per 90, Shots on Target (SoT) per 90, Take-Ons Attempted per 90, and Passes Completed Per 90.
For midfielders, the critical stats are Pass Completion Rate, Percentage of Tackles Won, Shot Creating Actions per 90, Distance Dribbled per 90, Goals per Shot, Interceptions Per 90, Total Progressive Pass Distance per 90, Progressive Passes Received Per 90, and the percentage of successful take-ons.
For defenders, the critical stats are the Percentage of Tackles Won, Shot Creating Actions per 90, non-penalty xG+AG per 90, Progressive Passes Per 90, Errors Per 90, Passes Completed Per 90, Dribble Distance per 90, Percentage of Passes between 15-30 Yards Completed, and Aerial Win Percentage.
For goalkeepers, the critical stats are Save Percentage, Post-Shot Expected Goals minus Goals Allowed (PSxG-GA) per 90, Pass Completion Rate, Errors Per 90, Percentage of Crosses Stopped, Average Pass Length, Goals allowed per shot on target (GA/SoT) per 90, Goals Allowed (GA) Per 90, and the Percentage of passes greater than 40 yards completed.
