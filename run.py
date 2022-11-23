from config import cfg
from flex_behavior.run import run_behavior_model, run_behavior_analyzer
from flex_community.run import run_community_model, run_community_analyzer
from flex_operation.run import run_operation_model, run_operation_analyzer


def run_flex_operation_model(config):
    operation_scenario_ids = [id_scenario for id_scenario in range(1, 97)]
    run_operation_model(operation_scenario_ids, config)


def run_flex_operation_analyzer(config):
    component_changes = [  # (str: component_name, int: start_state, int: end_state)
        ("ID_Building", 3, 2),
        ("ID_Building", 3, 1),
        ("ID_Building", 2, 1),
        ("ID_SEMS", 2, 1),
        ("ID_Boiler", 2, 1),
        ("ID_SpaceHeatingTank", 2, 1),
        ("ID_HotWaterTank", 2, 1),
        ("ID_PV", 2, 1),
        ("ID_Battery", 2, 1),
    ]
    components = [  # (str: component_name, int: amount_possible_states)
        ("ID_PV", 2),
        ("ID_SEMS", 2),
        ("ID_Boiler", 2),
        ("ID_SpaceHeatingTank", 2),
        ("ID_HotWaterTank", 2),
        ("ID_Battery", 2),
        ("ID_Building", 3),
    ]
    run_operation_analyzer(component_changes, components, config)


def run_flex_community_model(config):
    community_scenario_ids = [id_scenario for id_scenario in range(1, 11)]
    operation_scenario_ids = [id_scenario for id_scenario in range(1, 97)]
    run_community_model(
        community_scenario_ids=community_scenario_ids,
        operation_scenario_ids=operation_scenario_ids,
        config=config
    )


def run_flex_community_analyzer(config):
    community_scenario_ids = [id_scenario for id_scenario in range(1, 11)]
    run_community_analyzer(community_scenario_ids=community_scenario_ids, config=config)


def run_flex_behavior_model(config):
    behavior_scenario_ids = [id_scenario for id_scenario in range(1, 2)]
    run_behavior_model(behavior_scenario_ids, config)
    run_behavior_analyzer(behavior_scenario_ids, config)


def run_flex_behavior_analyzer(config):
    behavior_scenario_ids = [id_scenario for id_scenario in range(1, 2)]
    run_behavior_analyzer(behavior_scenario_ids, config)


if __name__ == "__main__":
    # run_flex_behavior_model(cfg)
    run_flex_behavior_analyzer(cfg)
    # run_flex_operation_model(cfg)
    # run_flex_operation_analyzer(cfg)
    # run_flex_community_model(cfg)
    # run_flex_community_analyzer(cfg)



