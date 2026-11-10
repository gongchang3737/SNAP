# -*- coding: utf-8 -*-
"""Fixed sensor budgets indexed by graph node count.

Edit ``SENSOR_BUDGETS`` to change the experiment budget mapping. Node counts
not listed here fail fast instead of silently falling back to a percentage.
"""


SENSOR_BUDGETS = {
    # Synthetic networks
    20: 3,
    50: 5,
    100: 10,
    200: 15,
    300: 20,
    500: 30,

    # Real-world networks
    11: 3,    # WDN_Net1
    36: 5,    # WDN_Net2
    97: 10,   # WDN_Net3
    14: 3,    # IEEE_14Bus
    30: 5,    # IEEE_30Bus
    57: 6,    # IEEE_57Bus
    118: 12,  # IEEE_118Bus
}


def get_sensor_budget(n_nodes, override=None):
    """Return and validate the sensor budget for ``n_nodes``."""
    n_nodes = int(n_nodes)
    budget = override if override is not None else SENSOR_BUDGETS.get(n_nodes)

    if budget is None:
        configured = ', '.join(str(n) for n in sorted(SENSOR_BUDGETS))
        raise ValueError(
            f'No sensor budget configured for n_nodes={n_nodes}. '
            f'Add it to SENSOR_BUDGETS in sensor_budget_config.py. '
            f'Configured node counts: {configured}'
        )

    budget = int(budget)
    if not 1 <= budget <= n_nodes:
        raise ValueError(
            f'Sensor budget must be between 1 and n_nodes; '
            f'got budget={budget}, n_nodes={n_nodes}'
        )
    return budget
