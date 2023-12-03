# AI_wargame_472
**Wargame AI project for COMP 472**

### Project description

This project builds upon the AI Wargame skeleton that was given, adding functionality for logging the game trace, taking input from the user for setting the game parameters, and validating and performing moves based on the game rules. Each of the team member contributed by engaging in group discussions about the project and coding the different requirements for deliverable 1.

The first deliverable allows human vs human gameplay, the AI logic will be implemented in the next installment. 

# How to run the Game:

Open up a terminal.The following command executes the game.

python ai_wargame.py 

**Note** Listed below is the list of command line arguments and their usage for setting the game parameters. If run without any arguments, the user will still be given the option to override the settings before starting the game.

usage: ai_wargame [-h] [--max_depth MAX_DEPTH] [--max_time MAX_TIME] [--max_turns MAX_TURNS] [--game_type GAME_TYPE] [--alpha_beta ALPHA_BETA] [--broker BROKER]

options:
  -h, --help            show this help message and exit
  --max_depth MAX_DEPTH
                        maximum search depth (default: None)
  --max_time MAX_TIME   maximum search time (default: 5.0)
  --max_turns MAX_TURNS
                        maximum number of turns (default: 100)
  --game_type GAME_TYPE
                        game type: auto|attacker|defender|manual (default: manual)
  --alpha_beta ALPHA_BETA
                        uses alpha-beta: True|False (default: True)
  --broker BROKER       play via a game broker (default: None)
