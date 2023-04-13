# Senior-Project
Senior Project files, code, notes

# Python Files

## Basic_Web_Scrap.py
  This file is to contain the functions meant to web scrap data for nhl teams.
  
  Functions:
  
    web_scrap()
      Pulls aggregated data from hockey reference for each team and outputs a dataframe.
      

## Games.py
  A file simply to be able to get a list of the games for the current dat
  
  Functions:
  
    todays_games()
      Pulls data and creates a list of nhl games for the current date


## Base.py
  A file that combines the functions web_scrap() abd todays_games() to create a baseline for model validation. This process involves scrapping aggregate data
  and creating new variables that are interpretable to a model.


## Probability.py
  A file designed to mimic the FiveThirtyEight model and generate probabilities from an ELO system and adjust them based off the results of the matches.
    

## ROC.py
  Produces ROC curves for the FiveThirtyEight model and baseline model.

## Test.py
  Playground for testing code.
