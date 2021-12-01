# Custom NLE NetHack Environment for OpenAI Gym to improve training
import gym
from gym.envs import registration
from nle.env import base
from nle import nethack
from nle.nethack import Command
import string


"""
CUSTOM TASKS - since we're updating the reward functions, this does really need to define new tasks.
"""
class NetHackBoost(base.NLE):
    # CommandAction
    COMMAND_ACTIONS = [
        Command.APPLY,
        Command.CAST,
        Command.CLOSE,
        Command.DIP,
        Command.DROP,
        Command.EAT,
        Command.ESC,
        Command.FIRE,
        Command.FORCE,
        Command.INVOKE,
        Command.KICK,
        Command.LOOT,
        Command.OFFER,
        Command.OPEN,
        Command.PAY,
        Command.PICKUP,
        Command.PRAY,
        Command.PUTON,
        Command.QUAFF,
        Command.QUIVER,
        Command.READ,
        Command.REMOVE,
        Command.RIDE,
        Command.RUB,
        Command.SEARCH,
        Command.TAKEOFF,
        Command.THROW,
        Command.TIP,
        Command.UNTRAP,
        Command.WEAR,
        Command.WIELD,
        Command.WIPE,
        Command.ZAP,
        nethack.TextCharacters.SPACE,
    ]

    # MiscDirection
    MISC_ACTIONS = list(nethack.MiscDirection)

    # Compass Directions
    COMPASS_ACTIONS = list(nethack.CompassDirection)

    # Raw Key Presses
    RAW_KEY_ACTIONS = [ord(x) for x in string.ascii_letters]

    # Reduced Action Space Only
    REDUCED_ACTIONS = tuple(COMPASS_ACTIONS + MISC_ACTIONS + COMMAND_ACTIONS)

    # Reduced Action Space and Menu Navigation
    REDUCED_ACTIONS_WITH_MENU = tuple(COMPASS_ACTIONS + MISC_ACTIONS + COMMAND_ACTIONS + RAW_KEY_ACTIONS)

    # Character strings
    CHARACTER_CODES = {
        'valkyrie-dwarf': 'val-dwa-law-fem',
        'wizard-elf': 'wiz-elf-cha-mal',
        'cavewoman': 'cav-hum-neu-fem',
        'ranger-elf': 'ran-elf-cha-mal',
    }

    # Create modified environment
    def __init__(self, character='valkyrie-dwarf', actions_mode='reduced'):
        super().__init__(
            character = NetHackBoost.CHARACTER_CODES[character],
            max_episode_steps = 60000,
            allow_all_yn_questions = True,
            actions = NetHackBoost.REDUCED_ACTIONS_WITH_MENU if actions_mode == 'with_menu' else NetHackBoost.REDUCED_ACTIONS,
        )