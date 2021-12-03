# Custom NLE NetHack Environment for OpenAI Gym to improve training
from nle.env import base
from nle import nethack
import string
import numpy as np


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

        # Initializes reward mode
        self.reward_mode = reward_mode

        # Intializes parent, gold task
        options = []
        if reward_mode != 'base':
            for option in nethack.NETHACKOPTIONS:
                if option.startswith("pickup_types"):
                    options.append("pickup_types:$")
                    continue
                options.append(option)

        super().__init__(
            character = NetHackBoost.CHARACTER_CODES[character],
            max_episode_steps = 60000,
            allow_all_yn_questions = True,
            actions = actions,
            options=options
        )
    
    def reset(self, *args, **kwargs):
        self.dungeon_explored = {}
        return super().reset(*args, **kwargs)

    def get_stair_reward(self, observation):
        stair_reward = 50.0 
        internal = observation[self._internal_index]
        stairs_down = internal[4]
        if stairs_down:
            return stair_reward
        return 0.0

    def get_gold_reward(self, last_observation, observation):
        passive_income_multiplier = 0.08
        
        old_blstats = last_observation[self._blstats_index]
        blstats = observation[self._blstats_index]

        old_gold = old_blstats[nethack.NLE_BL_GOLD]
        gold = blstats[nethack.NLE_BL_GOLD]

        return (gold - old_gold) + (gold - old_gold) ** (gold / old_gold + passive_income_multiplier)

    def get_time_penalty(self, last_observation, observation):
        penalty_step = -0.1

        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[nethack.NLE_BL_TIME]
        new_time = blstats_new[nethack.NLE_BL_TIME]

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0
        
        # Exponential penalty growth
        penalty = 2 ** self._frozen_steps * penalty_step
        return penalty
    
    def get_hunger_reward(self, last_observation, observation):
        old_internal = last_observation[self._internal_index]
        internal = observation[self._internal_index]

        old_uhunger = old_internal[7]
        uhunger = internal[7]

        reward = max(old_uhunger - uhunger, 0)
        return reward ** 2

    def get_explore_reward(self, observation):
        reward = 0
        glyphs = observation[self._glyph_index]
        blstats = observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        explored_old = 0

        if key in self.dungeon_explored:
            explored_old = self.dungeon_explored[key]
        reward = explored - explored_old
        self.dungeon_explored[key] = explored
        return reward * 2
    
    def get_healthy_reward(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_hp = blstats_old[nethack.NLE_BL_HP]
        new_hp = blstats_new[nethack.NLE_BL_HP]

        reward = max(new_hp - old_hp, -0.5)
        return reward

    # Create modified reward function
    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta, with added state loop"""
        if not self.env.in_normal_game():
            return 0.0

        # base reward function
        score_diff = super()._reward_fn(last_observation, action, observation, end_status)
        if self.reward_mode == 'base':
            return score_diff

        # enhanced reward function
        time_penalty = self.get_time_penalty(last_observation, observation)
        stair_reward = self.get_stair_reward(observation)
        gold_reward = self.get_gold_reward(last_observation, observation)
        hunger_reward = self.get_hunger_reward(last_observation, observation)
        explore_reward = self.get_explore_reward(observation)
        health_reward = self.get_healthy_reward(last_observation, observation)
        
        total_reward = sum(
            time_penalty,
            stair_reward,
            gold_reward,
            hunger_reward,
            explore_reward,
            health_reward,
        )
        return total_reward