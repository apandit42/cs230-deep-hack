# Custom NLE NetHack Environment for OpenAI Gym to improve training
import gym
from gym.envs import registration
from nle.env import base
from nle import nethack
from nle.nethack import Command
import string

"""
CONSTANT DEFINITIONS - This section defines the constants for our new tasks.
"""
# Compass Directions
COMPASS_ACTIONS = list(nethack.CompassDirection)

# MiscDirection
MISC_ACTIONS = list(nethack.MiscDirection)

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

# Raw Key Presses
RAW_KEY_ACTIONS = [ord(x) for x in string.string.ascii_letters]

# Total Reduced Action Space
ALLOWED_ACTIONS = tuple(COMPASS_ACTIONS + MISC_ACTIONS + COMMAND_ACTIONS + RAW_KEY_ACTIONS)

# Character strings
CHARACTERS = {
    'valkyrie-dwarf': 'val-dwa-law-fem',
    'wizard-elf': 'wiz-elf-cha-mal',
    'cavewoman': 'cav-hum-neu-fem',
    'ranger-elf': 'ran-elf-cha-mal',
}


"""
CUSTOM TASKS - since we're updating the reward functions, this does really need to define new tasks.
"""
class NetHackBoost(base.NLE):
    pass

"""
NEW SETTINGS WRAPPER - this wraps the updated settings we've enabled for NetHack.
"""