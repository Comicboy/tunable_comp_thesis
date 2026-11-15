"""LeRobot plugin: ACT with the stiffness as its own encoder token (policy type "act_ktoken").

LeRobot's ACT turns observation.state into ONE encoder token (a linear projection of all 48 numbers), so
log10 K is one input among 48 of that token. This variant splits it off: by default the transformer
encoder gets four tokens [latent, state without K (47), K (1), cube position (3)], i.e. ACT with only K
moved into a token of its own; with --policy.env_state_in_state_token=true the cube position joins the
state token (three tokens). Each token has its own projection and learned position embedding. Everything else is LeRobot's ACT unchanged (decoder, VAE, loss, temporal
ensembling, pre/post-processing).

Importing the package registers the type. Training (lerobot env, from the repo root):

    PYTHONPATH=scripts/il/lerobot_plugins lerobot-train --policy.discover_packages_path=act_ktoken \
        --policy.type=act_ktoken ...

policy_server.py and kswap_test.py import it themselves, so act_ktoken checkpoints load like ACT ones.
"""

from .configuration_act_ktoken import ACTKTokenConfig
from .modeling_act_ktoken import ACTKTokenPolicy
from .processor_act_ktoken import make_act_ktoken_pre_post_processors

__all__ = ["ACTKTokenConfig", "ACTKTokenPolicy", "make_act_ktoken_pre_post_processors"]