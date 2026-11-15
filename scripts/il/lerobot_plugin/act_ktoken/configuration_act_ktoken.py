"""Config of ACT with the stiffness as its own encoder token: LeRobot's ACTConfig plus the index of
log10 K in observation.state and whether the cube position joins the state token."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_ktoken")
@dataclass
class ACTKTokenConfig(ACTConfig):
    # Position of log10 K in observation.state (convert_to_lerobot.py puts it last: arm q, arm qdot,
    # hand q, fingertip forces, log10 K). Negative values count from the end.
    stiffness_index: int = -1
    # False (default, decided 2026-11-15 for a clean comparison with plain ACT, where only K moves): the cube
    # position (observation.environment_state) keeps its own token as in ACT, so the encoder gets 4 tokens
    # [latent, state without K, K, cube position]. True: the cube position joins the state token, 3 tokens
    # [latent, state without K + cube position, K]. The dataset is the same either way: LeRobot's ACT requires
    # an environment state (or images) among the inputs, so the cube stays a separate feature and is only
    # merged inside the model.
    env_state_in_state_token: bool = False

    def validate_features(self) -> None:
        super().validate_features()
        if self.robot_state_feature is None or self.robot_state_feature.shape[0] < 2:
            raise ValueError("act_ktoken needs observation.state with log10 K and at least one other entry.")
        n = self.robot_state_feature.shape[0]
        if not -n <= self.stiffness_index < n:
            raise ValueError(f"stiffness_index {self.stiffness_index} is outside observation.state ({n} entries).")