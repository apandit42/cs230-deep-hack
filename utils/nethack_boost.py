# Custom NLE NetHack Environment for OpenAI Gym to improve training
import gym
from gym.envs import registration
from nle.env import base
from nle import nethack
from nle.nethack import Command

# Reduced action space
ALLOWED_ACTIONS = tuple(
    list(nethack.CompassDirection)
    + list(nethack.MiscDirection)
    + [
        nethack.MiscAction.MORE,
        Command.APPLY,
        Command.CAST,
        Command.CLOSE,
        Command.DIP,
        Command.DROP,
        Command.EAT,
        Command.ESC,
        Command.FIRE,
        Command.FIGHT,
        Command.FORCE,
        Command.INVOKE,
        Command.JUMP,
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
        Command.SIT,
        Command.SWAP,
        Command.TAKEOFF,
        Command.TELEPORT,
        Command.THROW,
        Command.TIP,
        Command.TWOWEAPON,
        Command.UNTRAP,
        Command.WEAR,
        Command.WIELD,
        Command.WIPE,
        Command.ZAP,
        nethack.TextCharacters.SPACE
    ]
)