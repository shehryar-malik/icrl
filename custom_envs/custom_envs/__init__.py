from gym.envs.registration import register

ABS_PATH = "custom_envs.envs"

# =========================================================================== #
#                                   Cheetah                                   #
# =========================================================================== #

CHEETAH_LEN = 1000

register(
    id="HalfCheetahTest-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahTest",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="HCWithObstacle-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahWithObstacle",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="HCEqual-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahEqual",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="HCBackwards-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahBackward",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="HCWithPos-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahWithPos",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="HCWithPosTest-v0",
    entry_point=ABS_PATH+".half_cheetah:HalfCheetahWithPosTest",
    max_episode_steps=CHEETAH_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Walker                                    #
# =========================================================================== #

WALKER_LEN = 500

register(
    id="Walker2dTest-v0",
    entry_point=ABS_PATH+".walker:Walker2dTest",
    max_episode_steps=WALKER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="WalkerWithPos-v0",
    entry_point=ABS_PATH+".walker:WalkerWithPos",
    max_episode_steps=WALKER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="WalkerWithPosTest-v0",
    entry_point=ABS_PATH+".walker:WalkerWithPosTest",
    max_episode_steps=WALKER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                  Swimmer                                    #
# =========================================================================== #

SWIMMER_LEN = 500

register(
    id="SwimmerTest-v0",
    entry_point=ABS_PATH+".swimmer:SwimmerTest",
    max_episode_steps=SWIMMER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="SwimmerWithPos-v0",
    entry_point=ABS_PATH+".swimmer:SwimmerWithPos",
    max_episode_steps=SWIMMER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="SwimmerWithPosTest-v0",
    entry_point=ABS_PATH+".swimmer:SwimmerWithPosTest",
    max_episode_steps=SWIMMER_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                                   Point                                     #
# =========================================================================== #

POINT_LEN = 150

register(
    id="PointNullReward-v0",
    entry_point=ABS_PATH+".point:PointNullReward",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointNullRewardTest-v0",
    entry_point=ABS_PATH+".point:PointNullRewardTest",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointCircle-v0",
    entry_point=ABS_PATH+".point:PointCircle",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointCircleTest-v0",
    entry_point=ABS_PATH+".point:PointCircleTest",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointCircleTestBack-v0",
    entry_point=ABS_PATH+".point:PointCircleTestBack",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointTrack-v0",
    entry_point=ABS_PATH+".point:PointTrack",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="PointBridge-v0",
    entry_point=ABS_PATH+".point:PointBridge",
    max_episode_steps=POINT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)
# =========================================================================== #
#                                   Ant                                       #
# =========================================================================== #

ANT_LEN = 500

register(
    id="AntTest-v0",
    entry_point=ABS_PATH+".ant:AntTest",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntWall-v0",
    entry_point=ABS_PATH+".ant:AntWall",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntWallTest-v0",
    entry_point=ABS_PATH+".ant:AntWallTest",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntWallBroken-v0",
    entry_point=ABS_PATH+".ant:AntWallBroken",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntWallBrokenTest-v0",
    entry_point=ABS_PATH+".ant:AntWallBrokenTest",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntCircle-v0",
    entry_point=ABS_PATH+".ant:AntCircle",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="AntCircleTest-v0",
    entry_point=ABS_PATH+".ant:AntCircleTest",
    max_episode_steps=ANT_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                               Two Bridges                                   #
# =========================================================================== #

TWO_BRIDGES_LEN = 200

register(
    id="TwoBridges-v0",
    entry_point=ABS_PATH+".two_bridges:TwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="D2B-v0",
    entry_point=ABS_PATH+".two_bridges:DiscreteTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="DD2B-v0",
    entry_point=ABS_PATH+".two_bridges:DenseDiscreteTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="CDD2B-v0",
    entry_point=ABS_PATH+".two_bridges:ConstrainedDenseDiscreteTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="DDCDD2B-v0",
    entry_point=ABS_PATH+".two_bridges:DDConstrainedDenseDiscreteTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="C2B-v0",
    entry_point=ABS_PATH+".two_bridges:ContinuousTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="CC2B-v0",
    entry_point=ABS_PATH+".two_bridges:ConstrainedContinuousTwoBridges",
    max_episode_steps=TWO_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                               Three Bridges                                 #
# =========================================================================== #

THREE_BRIDGES_LEN = 200

register(
    id="ThreeBridges-v0",
    entry_point=ABS_PATH+".three_bridges:ThreeBridges",
    max_episode_steps=THREE_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="D3B-v0",
    entry_point=ABS_PATH+".three_bridges:DiscreteThreeBridges",
    max_episode_steps=THREE_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="DD3B-v0",
    entry_point=ABS_PATH+".three_bridges:DenseDiscreteThreeBridges",
    max_episode_steps=THREE_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="CDD3B-v0",
    entry_point=ABS_PATH+".three_bridges:ConstrainedDenseDiscreteThreeBridges",
    max_episode_steps=THREE_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="DDCDD3B-v0",
    entry_point=ABS_PATH+".three_bridges:DDConstrainedDenseDiscreteThreeBridges",
    max_episode_steps=THREE_BRIDGES_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

# =========================================================================== #
#                               Lap Grid World                                #
# =========================================================================== #

LAP_GRID_WORLD_LEN = 200

register(
    id="LGW-v0",
    entry_point=ABS_PATH+".lap_grid_world:LapGridWorld",
    max_episode_steps=LAP_GRID_WORLD_LEN,
    reward_threshold=None,
    nondeterministic=False,
)

register(
    id="CLGW-v0",
    entry_point=ABS_PATH+".lap_grid_world:ConstrainedLapGridWorld",
    max_episode_steps=LAP_GRID_WORLD_LEN,
    reward_threshold=None,
    nondeterministic=False,
)
