"""
Action State Machine for Embodied Manipulation

Defines the state transitions for Navigate, Pick, and Place actions.
Used to validate action sequences and compute transition rewards.
"""

from typing import List, Tuple, Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)



class State(Enum):
    """States in the manipulation state machine"""
    INIT = "INIT"                      # Initial state, nothing in hand
    AT_LOCATION = "AT_LOCATION"        # At a location, nothing in hand
    HOLDING = "HOLDING"                # Holding an object
    AT_LOCATION_HOLDING = "AT_LOCATION_HOLDING"  # At a location, holding an object


class Action(Enum):
    """Available actions"""
    NAVIGATE = "Navigate"
    PICK = "Pick"
    PLACE = "Place"
    UNKNOWN = "Unknown"


class ActionStateMachine:
    """
    State machine for embodied manipulation actions

    State Transition Rules:
        INIT ──Navigate──> AT_LOCATION
        AT_LOCATION ──Pick──> HOLDING
        AT_LOCATION ──Navigate──> AT_LOCATION
        HOLDING ──Navigate──> AT_LOCATION_HOLDING
        AT_LOCATION_HOLDING ──Navigate──> AT_LOCATION_HOLDING
        AT_LOCATION_HOLDING ──Place──> AT_LOCATION
        HOLDING ──Place──> INIT (if at same location where picked)
    """

    def __init__(self):
        # Valid transitions: (current_state, action) -> (next_state, reward)
        self.transitions = {
            # From INIT
            (State.INIT, Action.NAVIGATE): (State.AT_LOCATION, 1.0),
            (State.INIT, Action.PICK): (State.INIT, -1.0),      # Invalid: can't pick without navigating
            (State.INIT, Action.PLACE): (State.INIT, -1.0),     # Invalid: nothing to place

            # From AT_LOCATION
            (State.AT_LOCATION, Action.NAVIGATE): (State.AT_LOCATION, 0.8),  # Valid but potentially redundant
            (State.AT_LOCATION, Action.PICK): (State.HOLDING, 1.0),
            (State.AT_LOCATION, Action.PLACE): (State.AT_LOCATION, -1.0),    # Invalid: nothing to place

            # From HOLDING
            (State.HOLDING, Action.NAVIGATE): (State.AT_LOCATION_HOLDING, 1.0),
            (State.HOLDING, Action.PICK): (State.HOLDING, -1.0),             # Invalid: already holding
            (State.HOLDING, Action.PLACE): (State.INIT, 1.0),                # Valid: place at current location

            # From AT_LOCATION_HOLDING
            (State.AT_LOCATION_HOLDING, Action.NAVIGATE): (State.AT_LOCATION_HOLDING, 0.8),  # Valid but potentially redundant
            (State.AT_LOCATION_HOLDING, Action.PICK): (State.AT_LOCATION_HOLDING, -1.0),     # Invalid: already holding
            (State.AT_LOCATION_HOLDING, Action.PLACE): (State.AT_LOCATION, 1.0),
        }

    def parse_action(self, action_str: str) -> Tuple[Action, Optional[str]]:
        """
        Parse action string to Action enum and target

        Args:
            action_str: e.g., "Navigate(Table)" or "Pick(Apple)"

        Returns:
            (Action, target) tuple, e.g., (Action.NAVIGATE, "Table")
        """
        action_str = action_str.strip()

        if '(' not in action_str or ')' not in action_str:
            return Action.UNKNOWN, None

        try:
            action_name = action_str.split('(')[0].strip()
            target = action_str.split('(')[1].rstrip(')').strip()

            if action_name == "Navigate":
                return Action.NAVIGATE, target
            elif action_name == "Pick":
                return Action.PICK, target
            elif action_name == "Place":
                return Action.PLACE, target
            else:
                return Action.UNKNOWN, target
        except:
            return Action.UNKNOWN, None

    def step(self, current_state: State, action: Action) -> Tuple[State, float]:
        """
        Execute one state transition

        Args:
            current_state: Current state
            action: Action to execute

        Returns:
            (next_state, reward) tuple
        """
        transition = (current_state, action)

        if transition in self.transitions:
            next_state, reward = self.transitions[transition]
            return next_state, reward
        else:
            # Unknown transition, stay in current state with penalty
            return current_state, -1.0

    def validate_sequence(self, action_sequence: List[str]) -> Tuple[bool, float, List[Tuple[State, Action, float]]]:
        """
        Validate a complete action sequence

        Args:
            action_sequence: List of action strings, e.g., ["Navigate(Table)", "Pick(Apple)", ...]

        Returns:
            (is_valid, total_reward, transitions)
            - is_valid: True if no invalid transitions
            - total_reward: Sum of all transition rewards
            - transitions: List of (state, action, reward) for debugging
        """
        state = State.INIT
        total_reward = 0.0
        transitions = []
        is_valid = True

        for action_str in action_sequence:
            action, target = self.parse_action(action_str)

            if action == Action.UNKNOWN:
                is_valid = False
                transitions.append((state, action, -1.0))
                total_reward -= 1.0
                continue

            next_state, reward = self.step(state, action)
            transitions.append((state, action, reward))
            total_reward += reward

            if reward < 0:
                is_valid = False

            state = next_state

        return is_valid, total_reward, transitions

    def get_transition_rewards(self, action_sequence: List[str]) -> List[float]:
        """
        Get per-step transition rewards for an action sequence

        Args:
            action_sequence: List of action strings

        Returns:
            List of rewards for each transition
        """
        state = State.INIT
        rewards = []

        for action_str in action_sequence:
            action, _ = self.parse_action(action_str)
            next_state, reward = self.step(state, action)
            rewards.append(reward)
            state = next_state

        return rewards


def test_state_machine():
    """Test cases for state machine"""
    sm = ActionStateMachine()

    # Test case 1: Valid sequence
    valid_seq = ["Navigate(Table)", "Pick(Apple)", "Navigate(Basket)", "Place(Basket)"]
    is_valid, reward, transitions = sm.validate_sequence(valid_seq)
    logger.info(f"Test 1 - Valid sequence: {is_valid}, Reward: {reward:.2f}")
    for state, action, r in transitions:
        logger.info(f"  {state.value} --{action.value}--> (reward: {r:.2f})")

    logger.info()

    # Test case 2: Invalid sequence (Pick before Navigate)
    invalid_seq1 = ["Pick(Apple)", "Navigate(Table)", "Place(Basket)"]
    is_valid, reward, transitions = sm.validate_sequence(invalid_seq1)
    logger.info(f"Test 2 - Pick before Navigate: {is_valid}, Reward: {reward:.2f}")
    for state, action, r in transitions:
        logger.info(f"  {state.value} --{action.value}--> (reward: {r:.2f})")

    logger.info()

    # Test case 3: Invalid sequence (Place without Pick)
    invalid_seq2 = ["Navigate(Table)", "Place(Basket)"]
    is_valid, reward, transitions = sm.validate_sequence(invalid_seq2)
    logger.info(f"Test 3 - Place without Pick: {is_valid}, Reward: {reward:.2f}")
    for state, action, r in transitions:
        logger.info(f"  {state.value} --{action.value}--> (reward: {r:.2f})")

    logger.info()

    # Test case 4: Redundant Navigate
    redundant_seq = ["Navigate(Table)", "Navigate(Table)", "Pick(Apple)", "Navigate(Basket)", "Place(Basket)"]
    is_valid, reward, transitions = sm.validate_sequence(redundant_seq)
    logger.info(f"Test 4 - Redundant Navigate: {is_valid}, Reward: {reward:.2f}")
    for state, action, r in transitions:
        logger.info(f"  {state.value} --{action.value}--> (reward: {r:.2f})")


if __name__ == "__main__":
    test_state_machine()
