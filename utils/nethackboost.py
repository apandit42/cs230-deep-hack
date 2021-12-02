# Custom NLE NetHack Environment for OpenAI Gym to improve training
from nle.env import base
from nle import nethack
import string


"""
CUSTOM TASKS - since we're updating the reward functions, this does really need to define new tasks.
"""
class NetHackBoost(base.NLE):
    # CommandAction
    COMMAND_ACTIONS = [
        nethack.Command.APPLY,
        nethack.Command.CAST,
        nethack.Command.CLOSE,
        nethack.Command.DIP,
        nethack.Command.DROP,
        nethack.Command.EAT,
        nethack.Command.ESC, # leave menu
        nethack.Command.FIRE,
        nethack.Command.FORCE,
        nethack.Command.INVOKE,
        nethack.Command.KICK,
        nethack.Command.LOOT,
        nethack.Command.OFFER,
        nethack.Command.OPEN,
        nethack.Command.PAY,
        nethack.Command.PICKUP,
        nethack.Command.PRAY,
        nethack.Command.PUTON,
        nethack.Command.QUAFF,
        nethack.Command.QUIVER,
        nethack.Command.READ,
        nethack.Command.REMOVE,
        nethack.Command.RIDE,
        nethack.Command.RUB,
        nethack.Command.SEARCH,
        nethack.Command.TAKEOFF,
        nethack.Command.THROW,
        nethack.Command.TIP,
        nethack.Command.UNTRAP,
        nethack.Command.WEAR,
        nethack.Command.WIELD,
        nethack.Command.WIPE,
        nethack.Command.ZAP,
        nethack.TextCharacters.DOLLAR,
        nethack.TextCharacters.SPACE, # confirm selection
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

    # Base Actions
    BASE_ACTIONS = base.FULL_ACTIONS

    # Character strings
    CHARACTER_CODES = {
        'valkyrie-dwarf': 'val-dwa-law-fem',
        'wizard-elf': 'wiz-elf-cha-mal',
        'cavewoman': 'cav-hum-neu-fem',
        'ranger-elf': 'ran-elf-cha-mal',
    }

    # Create modified environment
    def __init__(self, character='valkyrie-dwarf', actions_mode='reduced', reward_mode='base'):
        # Initialize actions list, which by default is None and uses 79 actions NLE defines as "useful"
        actions = None
        if actions_mode == 'reduced':
            actions = NetHackBoost.REDUCED_ACTIONS
        elif actions_mode == 'with_menu':
            actions = NetHackBoost.REDUCED_ACTIONS_WITH_MENU
        self.actions_list = actions
        super().__init__(
            character = NetHackBoost.CHARACTER_CODES[character],
            max_episode_steps = 60000,
            allow_all_yn_questions = True,
            actions = actions,
        )