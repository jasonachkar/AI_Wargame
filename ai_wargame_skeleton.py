from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import math
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

class MoveDirection(Enum):
    Up = 0
    Left = 1
    Down = 2
    Right = 3

class EvaluationType(Enum):
    E0 = 0
    E1 = 1
    E2 = 2

##############################################################################################################

@dataclass(slots=True)
class Logger:
    game_trace: str = ''
    
    def log(self, text: str):
        self.game_trace += text + '\n'
        
    def write_to_file(self, alpha_beta : bool, timeout : int, max_turns : int):
        with open(f"gameTrace-{alpha_beta}-{timeout}-{max_turns}.txt", 'w') as f:
            f.write(self.game_trace)

    def write_to_console(self):
        print(self.game_trace)
            
##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"
    
    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()
    
    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()
    
    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()
    
    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()
    
    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))
    
    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))
    
    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    e_function : EvaluationType = EvaluationType.E0

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    evaluations : int = 0
    branching_factors : list[int] = field(default_factory=list)
    total_seconds: float = 0.0

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    _time_has_elapsed : bool = False
    logger : Logger = field(default_factory=Logger)
    eval_type : EvaluationType = EvaluationType.E0

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        self.init_stats()
        self.eval_type = self.options.e_function
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def init_stats(self) -> None:
        for i in range(1, self.options.max_depth +1):
            self.stats.evaluations_per_depth[i] = 0
        self.stats.branching_factors = []

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit_src = self.get(coords.src)
        if unit_src is None or unit_src.player != self.next_player:
            return False
        unit_dst = self.get(coords.dst)
        if (unit_src == unit_dst): 
            return True
        if unit_dst is None:
            return self.is_valid_movement(unit_src, coords.src, coords.dst)
        if unit_dst.player != self.next_player:
            return self.is_valid_attack(coords.src, coords.dst)
        return self.is_valid_repair(unit_src, unit_dst, coords.src, coords.dst)
    
    def is_valid_movement(self, unit_src: Unit, src: Coord, dst: Coord) -> bool:
        """ Determine whether the move is a valid movement. Assumes the destination is free."""
        direction = None
        for i, adj in enumerate(src.iter_adjacent()):
            if adj == dst:
                direction = MoveDirection(i)
            elif (self.get(adj) is not None and self.get(adj).player != self.next_player and 
                  unit_src.type != UnitType.Tech and unit_src.type != UnitType.Virus):
                return False
        return (direction is not None and 
                ((unit_src.player == Player.Attacker and (direction == MoveDirection.Up or direction == MoveDirection.Left)) or 
                 (unit_src.player == Player.Defender and (direction == MoveDirection.Down or direction == MoveDirection.Right)) or
                  unit_src.type == UnitType.Tech or unit_src.type == UnitType.Virus))
    
    def is_valid_attack(self, src: Coord, dst: Coord) -> bool:
        """ Determine whether the move is a valid attack. Assumes the destination is occupied by adversary unit."""
        for adj in src.iter_adjacent():
            if adj == dst:
                return True
        return False
     
            
    def is_valid_repair(self, unit_src: Unit, unit_dst: Unit, src: Coord, dst: Coord) -> bool:
        """ Determine whether the move is a valid repair. Assumes the destination is occupied by friendly unit."""
        isAdjacent = False
        for adj in src.iter_adjacent():
            if adj == dst:
                isAdjacent = True
        return isAdjacent and unit_src.repair_amount(unit_dst) > 0 

    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair."""
        if self.is_valid_move(coords):
            unit_dst = self.get(coords.dst)
            unit_src = self.get(coords.src)
            message = ""
            if (coords.src == coords.dst):
                for adj in coords.src.iter_range(1):
                    if self.get(adj) is not None:
                        self.mod_health(adj, -2)
                self.mod_health(coords.src, -9)
                message = f"Self-destruct at {coords.src}"
            elif (unit_dst is None):
                self.set(coords.dst,unit_src)
                self.set(coords.src,None)
                message = f"Move from {coords.src} to {coords.dst}"
            elif (unit_dst.player != self.next_player):
                self.mod_health(coords.dst, -unit_src.damage_amount(unit_dst))
                self.mod_health(coords.src, -unit_dst.damage_amount(unit_src))
                message = f"Attack from {coords.src} to {coords.dst}"
            else:
                self.mod_health(coords.dst, unit_src.repair_amount(unit_dst))
                message = f"Repair from {coords.src} to {coords.dst}"
            return (True,message)
        return (False,"invalid move")
    
    def perform_barebones_move(self, coords : CoordPair) -> None:
        """Assumes valid move. Perform a move expressed as a CoordPair. No message,no validation."""
        unit_dst = self.get(coords.dst)
        unit_src = self.get(coords.src)
        if (coords.src == coords.dst):
            for adj in coords.src.iter_range(1):
                if self.get(adj) is not None:
                    self.mod_health(adj, -2)
            self.mod_health(coords.src, -9)
        elif (unit_dst is None):
            self.set(coords.dst,unit_src)
            self.set(coords.src,None)
        elif (unit_dst.player != self.next_player):
            self.mod_health(coords.dst, -unit_src.damage_amount(unit_dst))
            self.mod_health(coords.src, -unit_dst.damage_amount(unit_src))
        else:
            self.mod_health(coords.dst, unit_src.repair_amount(unit_dst))

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()
    
    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')
    
    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    self.next_turn()
                    self.logger.log(result)
                    break
                else:
                    print("The move is not valid! Try again.")
                    self.logger.log(f"The move {mv} is not valid! Try again.\n")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def player_ai(self, player: Player) -> Tuple[Coord, Unit]:
        """Get player's AI position."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.type == UnitType.AI and unit.player == player:
                return (coord, unit)
            
    def is_finished(self) -> bool:
        """Check if the game is over."""
        return (not self._attacker_has_ai) or (not self._defender_has_ai) 

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self._time_has_elapsed:
            return self.next_player
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker    
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()
    
    def eval_f(self, currentState: Game) -> float:
        player = self.next_player
        opponent = self.next_player.next()
        value=0
        match self.eval_type:
            case EvaluationType.E1:
                atk_v_coord = None
                atk_f_coord = None
                atk_p_coord = None
                def_f_coord = None
                def_p_coord = None
                def_t_coord = None

                for (coord,unit) in currentState.player_units(Player.Attacker):
                    if unit.type == UnitType.Virus:
                        atk_v_coord = coord
                    elif unit.type == UnitType.Firewall:
                        atk_f_coord = coord
                    elif unit.type == UnitType.Program:
                        atk_p_coord = coord

                for (coord,unit) in currentState.player_units(Player.Defender):
                    if unit.type == UnitType.Firewall:
                        def_f_coord = coord
                    elif unit.type == UnitType.Program:
                        def_p_coord = coord
                    elif unit.type == UnitType.Tech:
                        def_t_coord = coord
                return ((100 * self.manhattan_dist(atk_v_coord, def_f_coord) +
                        10 * self.manhattan_dist(atk_f_coord, def_p_coord) +
                        self.manhattan_dist(atk_p_coord, def_t_coord)) / 3)
            case EvaluationType.E2:
                p_ai_coords, p_ai_unit = currentState.player_ai(player)
                o_ai_coords, o_ai_unit = currentState.player_ai(opponent)
                for (coord,unit) in currentState.player_units(player):
                    value += unit.damage_amount(o_ai_unit)*(1 / currentState.manhattan_dist(coord, o_ai_coords))
                for (coord, unit) in currentState.player_units(opponent):
                    value -= unit.damage_amount(p_ai_unit)*(1 / currentState.manhattan_dist(coord, p_ai_coords))
                return value
            case _:
                for (_,unit) in currentState.player_units(player):
                    value = value + 9999 if unit.type == UnitType.AI else value + 3
                for (_,unit) in currentState.player_units(opponent):
                    value = value - 9999 if unit.type == UnitType.AI else value - 3
                return value
    
    def manhattan_dist(self, src: Coord, dst: Coord) ->int:
        if src is None or dst is None:
            return 0
        return abs((src.row-dst.row))+abs((src.col-dst.col))
    
    def minimax_init(self, start_time: datetime) -> Tuple[int, CoordPair | None]:
        # if alpha_beta then use alphabeta pruning
        if self.options.alpha_beta:
            return self.alphabeta(self.clone(), 0, -math.inf, math.inf, True, start_time)
        
        # else, regular minimax
        return self.minimax(self.clone(), 0, True, start_time)

    # regular minimax function
    def minimax(self, currentState: Game, depth: int, isMax: bool, start_time: datetime) -> Tuple[int, CoordPair | None]:
        if (depth == self.options.max_depth or currentState.is_finished() or (datetime.now() - start_time).total_seconds() > self.options.max_time-.2):
            if (currentState.is_finished()):
                if (self.next_player == Player.Attacker):
                    if (not currentState._attacker_has_ai):
                        return MIN_HEURISTIC_SCORE, None
                    else:
                        return MAX_HEURISTIC_SCORE, None
                else:
                    if (not currentState._defender_has_ai):
                        return MIN_HEURISTIC_SCORE, None
                    else:
                        return MAX_HEURISTIC_SCORE, None
            self.stats.evaluations += 1
            self.stats.evaluations_per_depth[depth] += 1
            return self.eval_f(currentState), None
        moves = list(currentState.move_candidates())
        self.stats.branching_factors.append(len(moves))
        
        if isMax:
            current_max = (-math.inf, None)
            for move in moves:
                currentGame = currentState.clone()
                currentGame.perform_barebones_move(move)
                currentGame.next_turn()
                min_tuple = self.minimax(currentGame, depth+1, False, start_time)
                if min_tuple[0] > current_max[0]:
                    current_max = (min_tuple[0],move)
            return current_max
        else:  
            current_min = (math.inf, None)
            for move in moves:
                currentGame = currentState.clone()
                currentGame.perform_barebones_move(move)
                currentGame.next_turn()
                max_tuple = self.minimax(currentGame, depth+1, True, start_time)
                if max_tuple[0] < current_min[0]:
                    current_min = (max_tuple[0],move)
            return current_min

    def alphabeta(self, currentState: Game, depth: int, alpha: int, beta: int, isMax: bool,start_time: datetime) -> Tuple[int, CoordPair | None]:
        if (depth == self.options.max_depth or currentState.is_finished() or (datetime.now() - start_time).total_seconds() > self.options.max_time-.2):
            if (currentState.is_finished()):
                if (self.next_player == Player.Attacker):
                    if (not currentState._attacker_has_ai):
                        return MIN_HEURISTIC_SCORE, None
                    else:
                        return MAX_HEURISTIC_SCORE, None
                else:
                    if (not currentState._defender_has_ai):
                        return MIN_HEURISTIC_SCORE, None
                    else:
                        return MAX_HEURISTIC_SCORE, None
            self.stats.evaluations += 1
            self.stats.evaluations_per_depth[depth] += 1
            return self.eval_f(currentState), None
        moves = list(currentState.move_candidates())
        self.stats.branching_factors.append(len(moves))
        
        if isMax:
            current_max = (-math.inf, None)
            for move in moves:
                currentGame = currentState.clone()
                currentGame.perform_barebones_move(move)
                currentGame.next_turn()
                min_tuple = self.alphabeta(currentGame, depth+1, alpha, beta, False, start_time)
                if min_tuple[0] > current_max[0]:
                    current_max = (min_tuple[0],move)
                alpha = max(alpha,min_tuple[0])
                if beta <= alpha:
                    break
            return current_max
        else:  
            current_min = (math.inf, None)
            for move in moves:
                currentGame = currentState.clone()
                currentGame.perform_barebones_move(move)
                currentGame.next_turn()
                max_tuple = self.alphabeta(currentGame, depth+1, alpha, beta, True, start_time)
                if max_tuple[0] < current_min[0]:
                    current_min = (max_tuple[0],move)
                beta = min(beta,max_tuple[0])
                if beta <= alpha:
                    break
            return current_min

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta."""
        start_time = datetime.now()
        (score, move) = self.minimax_init(start_time)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        self._time_has_elapsed = elapsed_seconds > self.options.max_time
        print(f"Heuristic score: {score}")
        self.logger.log(f"Heuristic score: {score}")
        print(f"Evals per depth: ",end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        self.logger.log(f"Elapsed time: {elapsed_seconds:0.1f}s")
        self.logger.log(f"Cumulative evaluations: {self.stats.evaluations}")
        self.logger.log(f"Average branching factor: {self.average_branching_factor():.2f}")
        self.logger.log(f"{self.evals_depth_stats()}")
        return move
        
    def average_branching_factor(self) -> int:
        branching_factors_sum = 0
        for bf in self.stats.branching_factors:
            branching_factors_sum += bf
        return branching_factors_sum / len(self.stats.branching_factors)
        
    def evals_depth_stats(self) -> str:
        s1 = "Cumulative evals by depth: "
        s2 = "Cumulative % evals by depth: "
        if self.stats.evaluations > 0:
            for k in self.stats.evaluations_per_depth:
                s1 += f"{k}={self.stats.evaluations_per_depth[k]} "
                s2 += f"{k}={self.stats.evaluations_per_depth[k]/self.stats.evaluations:.2%} "
        return s1 + "\n" + s2 + "\n"

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, default=5.0, help='maximum search time')
    parser.add_argument('--max_turns', type=int, default=100, help='maximum number of turns')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--alpha_beta', type=str, default="True", help='uses alpha-beta: True|False')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--e_function', type=str, default="e0", help='evaluation function: e0|e1|e2')
    args = parser.parse_args()
    
    # allows the user to modify game parameters
    answer_gametype = input(f"\nThe current Game Type is: {args.game_type}. Would you like to change the Game Type Y/N: ")
    answer_gametype = answer_gametype.lower()
    if answer_gametype == "y":
        args.game_type = input("Choose any of the following game types (H-H, H-AI, AI-H and AI-AI): ")

    answer_max = input(f"\nThe current maximum allowed time for your program to return a move is: {args.max_time} seconds."
                       " Would you like to modify this parameter Y/N: ")
    answer_max = answer_max.lower()
    if answer_max == "y":
        args.max_time = float(input("Enter the maximum search time (seconds): "))
    
    answer_maxturns = input(f"\nThe current maximum number of turns is {args.max_turns}. Would you like to modify this? Y/N: ")
    answer_maxturns=answer_maxturns.lower()
    if answer_maxturns=="y":
        args.max_turns = int(input("Please enter the maximum number of turns: "))

    answer_alpha = input(f"\nThe current Alpha-Beta setting is: {args.alpha_beta}. Would you like to modify the Alpha-Beta parameter? Y/N: ")
    answer_alpha = answer_alpha.lower()
    if answer_alpha == "y":
        args.alpha_beta = input("For Alpha-Beta On (Enter True)| Off (Enter False): ")

    answer_e = input(f"\nThe current evaluation function is: {args.e_function}. Would you like to modify the evaluation function? Y/N: ")
    answer_e = answer_e.lower()
    if answer_e == "y":
        args.e_function = input("Enter e0 or e1 or e2: ")

    # parse the game type
    if args.game_type == "attacker" or args.game_type=="H-AI":
        game_type = GameType.AttackerVsComp
    elif args.game_type == "defender" or args.game_type=="AI-H":
        game_type = GameType.CompVsDefender
    elif args.game_type == "manual" or args.game_type=="H-H":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # parse the evaluation function
    if args.e_function == "e1":
        options.e_function = EvaluationType.E1
    elif args.e_function == "e2":
        options.e_function = EvaluationType.E2
    else:
        options.e_function = EvaluationType.E0 

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker
    if args.max_turns is not None:
        options.max_turns = args.max_turns
    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta.lower().startswith("t")

    # create a new game
    game = Game(options=options)
    
    game_type_representation = ''
    if game.options.game_type is GameType.AttackerVsDefender:
        game_type_representation = 'P1 (Human) vs. P2 (Human)'
    elif game.options.game_type is GameType.AttackerVsComp:
        game_type_representation = 'P1 (Human) vs. P2 (AI)'
    elif game.options.game_type is GameType.CompVsDefender:
        game_type_representation = 'P1 (AI) vs. P2 (Human)'
    else:
        game_type_representation = 'P1 (AI) vs. P2 (AI)'
    
    options_trace = (
                    f'Timeout: {game.options.max_time} seconds\n'
                    f'Maximum number of turns: {game.options.max_turns} turns\n'
                    f'Alpha-Beta: {game.options.alpha_beta}\n'
                    f'Game Type: {game_type_representation}\n'
                    f'Heuristic: {game.options.e_function}'
                    )
    
    game.logger.log("-------------\n\nGame Parameters:")
    game.logger.log(options_trace)
    game.logger.log("\n-------------\n\nGame Play:")
    game.logger.write_to_console()
                
    # the main game loop
    while True:
        print()
        print(game)
        game.logger.log(str(game))
        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            game.logger.log(f"{winner.name} wins in {game.turns_played} turns!")
            game.logger.write_to_file(game.options.alpha_beta, game.options.max_time, game.options.max_turns)
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                exit(1)

##############################################################################################################

if __name__ == '__main__':
    main()
