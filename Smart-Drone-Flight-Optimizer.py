"""
Semester 3  Project

Features:
1. Grid-based environment
2. Obstacles / no-fly zones
3. Safety buffer around obstacles
4. A* path planning
5. Dijkstra path planning
6. Route comparison
7. Drone performance calculations
8. Battery-energy estimation
9. Wind penalty model
10. Weighted route cost
11. Streamlit dashboard
12. CSV export


"""


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
import math
import time


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Smart Drone Flight Optimizer",
    page_icon="🛩️",
    layout="wide"
)

st.title("🛩️ Smart Drone Flight Path Optimizer")

st.write(
    "Create your own obstacle map and calculate a safe drone route."
)


# ============================================================
# GRID SETTINGS
# ============================================================

ROWS = 20
COLS = 20


# ============================================================
# SESSION STATE
# ============================================================

if "obstacle_df" not in st.session_state:

    initial_grid = np.zeros(
        (ROWS, COLS),
        dtype=bool
    )

    st.session_state.obstacle_df = pd.DataFrame(
        initial_grid,
        index=[f"R{i}" for i in range(ROWS)],
        columns=[f"C{i}" for i in range(COLS)]
    )

if "start" not in st.session_state:
    st.session_state.start = (0, 0)

if "goal" not in st.session_state:
    st.session_state.goal = (19, 19)

if "route" not in st.session_state:
    st.session_state.route = None


# ============================================================
# A* HEURISTIC
# ============================================================

def heuristic(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


# ============================================================
# NEIGHBORS
# ============================================================

def get_neighbors(node, grid):

    directions = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),

        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1)
    ]

    neighbors = []

    r, c = node

    for dr, dc in directions:

        nr = r + dr
        nc = c + dc

        if (
            0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
        ):

            if grid[nr, nc] == 0:

                neighbors.append(
                    (nr, nc)
                )

    return neighbors


# ============================================================
# MOVEMENT COST
# ============================================================

def movement_cost(a, b):

    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])

    if dr == 1 and dc == 1:
        return math.sqrt(2)

    return 1.0


# ============================================================
# RECONSTRUCT PATH
# ============================================================

def reconstruct_path(came_from, current):

    path = [current]

    while current in came_from:

        current = came_from[current]

        path.append(current)

    path.reverse()

    return path


# ============================================================
# A* ALGORITHM
# ============================================================

def a_star(grid, start, goal):

    if grid[start] == 1:
        return None, 0

    if grid[goal] == 1:
        return None, 0

    open_set = []

    counter = 0

    heapq.heappush(
        open_set,
        (0, counter, start)
    )

    came_from = {}

    g_score = {
        start: 0.0
    }

    visited = set()

    nodes_explored = 0

    while open_set:

        _, _, current = heapq.heappop(
            open_set
        )

        if current in visited:
            continue

        visited.add(current)

        nodes_explored += 1

        if current == goal:

            return (
                reconstruct_path(
                    came_from,
                    current
                ),
                nodes_explored
            )

        for neighbor in get_neighbors(
            current,
            grid
        ):

            if neighbor in visited:
                continue

            new_cost = (
                g_score[current]
                + movement_cost(
                    current,
                    neighbor
                )
            )

            if (
                neighbor not in g_score
                or new_cost < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = new_cost

                f_score = (
                    new_cost
                    + heuristic(
                        neighbor,
                        goal
                    )
                )

                counter += 1

                heapq.heappush(
                    open_set,
                    (
                        f_score,
                        counter,
                        neighbor
                    )
                )

    return None, nodes_explored


# ============================================================
# SAFETY BUFFER
# ============================================================

def create_safety_buffer(
    grid,
    buffer_size
):

    result = grid.copy()

    obstacle_locations = np.argwhere(
        grid == 1
    )

    for r, c in obstacle_locations:

        for dr in range(
            -buffer_size,
            buffer_size + 1
        ):

            for dc in range(
                -buffer_size,
                buffer_size + 1
            ):

                nr = r + dr
                nc = c + dc

                if (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                ):

                    result[nr, nc] = 1

    return result


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(
    path,
    cell_size
):

    if path is None:
        return 0.0

    distance = 0.0

    for a, b in zip(
        path[:-1],
        path[1:]
    ):

        dr = abs(
            b[0] - a[0]
        )

        dc = abs(
            b[1] - a[1]
        )

        if dr == 1 and dc == 1:

            distance += (
                cell_size * math.sqrt(2)
            )

        else:

            distance += cell_size

    return distance


# ============================================================
# DRAW MAP
# ============================================================

def draw_map(
    obstacle_grid,
    start,
    goal,
    route=None,
    safety_grid=None
):

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    # --------------------------
    # Base map
    # --------------------------

    ax.imshow(
        obstacle_grid,
        cmap="Greys",
        interpolation="nearest"
    )

    # --------------------------
    # Safety buffer
    # --------------------------

    if safety_grid is not None:

        buffer_only = (
            (safety_grid == 1)
            &
            (obstacle_grid == 0)
        )

        if np.any(buffer_only):

            buffer_mask = np.ma.masked_where(
                ~buffer_only,
                buffer_only
            )

            ax.imshow(
                buffer_mask,
                cmap="Oranges",
                alpha=0.35
            )

    # --------------------------
    # Route
    # --------------------------

    if route is not None:

        route_array = np.array(
            route
        )

        ax.plot(
            route_array[:, 1],
            route_array[:, 0],
            linewidth=3,
            label="A* Route"
        )

    # --------------------------
    # Start
    # --------------------------

    ax.scatter(
        start[1],
        start[0],
        s=250,
        marker="o",
        label="START"
    )

    # --------------------------
    # Goal
    # --------------------------

    ax.scatter(
        goal[1],
        goal[0],
        s=300,
        marker="*",
        label="TARGET"
    )

    ax.set_title(
        "Custom Drone Environment"
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    ax.set_xticks(
        np.arange(
            0,
            COLS,
            1
        )
    )

    ax.set_yticks(
        np.arange(
            0,
            ROWS,
            1
        )
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🛠️ Create Your Environment"
)

st.sidebar.write(
    "Tick cells in the grid to create obstacles."
)


# ============================================================
# START POINT
# ============================================================

st.sidebar.subheader(
    "📍 Start Point"
)

start_row = st.sidebar.number_input(
    "Start Row",
    min_value=0,
    max_value=ROWS - 1,
    value=0,
    step=1
)

start_col = st.sidebar.number_input(
    "Start Column",
    min_value=0,
    max_value=COLS - 1,
    value=0,
    step=1
)


# ============================================================
# TARGET POINT
# ============================================================

st.sidebar.subheader(
    "🎯 Target Point"
)

goal_row = st.sidebar.number_input(
    "Target Row",
    min_value=0,
    max_value=ROWS - 1,
    value=19,
    step=1
)

goal_col = st.sidebar.number_input(
    "Target Column",
    min_value=0,
    max_value=COLS - 1,
    value=19,
    step=1
)


# ============================================================
# SAFETY BUFFER
# ============================================================

st.sidebar.subheader(
    "🛡️ Safety"
)

safety_buffer = st.sidebar.slider(
    "Safety Buffer (cells)",
    min_value=0,
    max_value=3,
    value=1
)


# ============================================================
# CELL SIZE
# ============================================================

cell_size = st.sidebar.number_input(
    "Grid Cell Size (meters)",
    min_value=1.0,
    max_value=1000.0,
    value=10.0,
    step=1.0
)


# ============================================================
# DRONE DATA
# ============================================================

st.sidebar.subheader(
    "🛩️ Drone Parameters"
)

drone_speed = st.sidebar.number_input(
    "Cruise Speed (m/s)",
    min_value=0.5,
    max_value=100.0,
    value=10.0,
    step=0.5
)

battery_capacity = st.sidebar.number_input(
    "Battery (mAh)",
    min_value=100.0,
    max_value=50000.0,
    value=5000.0,
    step=100.0
)

voltage = st.sidebar.number_input(
    "Battery Voltage (V)",
    min_value=3.0,
    max_value=100.0,
    value=14.8,
    step=0.1
)

power = st.sidebar.number_input(
    "Average Power (W)",
    min_value=10.0,
    max_value=5000.0,
    value=200.0,
    step=10.0
)


# ============================================================
# WIND
# ============================================================

st.sidebar.subheader(
    "💨 Wind"
)

wind_speed = st.sidebar.number_input(
    "Wind Speed (m/s)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.5
)

wind_direction = st.sidebar.slider(
    "Wind Direction (°)",
    min_value=0,
    max_value=359,
    value=0
)


# ============================================================
# APPLY START / TARGET
# ============================================================

if st.sidebar.button(
    "✅ Apply Start & Target",
    use_container_width=True
):

    new_start = (
        int(start_row),
        int(start_col)
    )

    new_goal = (
        int(goal_row),
        int(goal_col)
    )

    grid_now = (
        st.session_state.obstacle_df.values
        .astype(int)
    )

    if grid_now[new_start] == 1:

        st.sidebar.error(
            "Start point is inside an obstacle."
        )

    elif grid_now[new_goal] == 1:

        st.sidebar.error(
            "Target point is inside an obstacle."
        )

    else:

        st.session_state.start = new_start
        st.session_state.goal = new_goal
        st.session_state.route = None

        st.sidebar.success(
            "Start and target updated."
        )


# ============================================================
# RESET
# ============================================================

if st.sidebar.button(
    "🔄 Clear All Obstacles",
    use_container_width=True
):

    st.session_state.obstacle_df = pd.DataFrame(
        np.zeros(
            (ROWS, COLS),
            dtype=bool
        ),
        index=[
            f"R{i}"
            for i in range(ROWS)
        ],
        columns=[
            f"C{i}"
            for i in range(COLS)
        ]
    )

    st.session_state.route = None

    st.rerun()


# ============================================================
# MAP EDITOR
# ============================================================

st.subheader(
    "🧱 MANUAL OBSTACLE EDITOR"
)

st.write(
    "Change any cell from unchecked to checked to make it an obstacle."
)

edited_df = st.data_editor(
    st.session_state.obstacle_df,
    use_container_width=True,
    height=520,
    key="map_editor"
)

st.session_state.obstacle_df = edited_df


# ============================================================
# BUILD GRID
# ============================================================

grid = (
    st.session_state.obstacle_df.values
    .astype(int)
)

start = st.session_state.start
goal = st.session_state.goal


# ============================================================
# PREVENT START / TARGET AS OBSTACLE
# ============================================================

grid[start] = 0
grid[goal] = 0


# ============================================================
# SAFETY GRID
# ============================================================

safety_grid = create_safety_buffer(
    grid,
    safety_buffer
)

safety_grid[start] = 0
safety_grid[goal] = 0


# ============================================================
# PREVIEW MAP
# ============================================================

st.subheader(
    "🗺️ Your Map"
)

preview_fig = draw_map(
    grid,
    start,
    goal,
    route=None,
    safety_grid=safety_grid
)

st.pyplot(
    preview_fig,
    use_container_width=True
)

plt.close(
    preview_fig
)


# ============================================================
# CURRENT MAP INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Obstacles",
    int(np.sum(grid == 1))
)

col2.metric(
    "Start",
    str(start)
)

col3.metric(
    "Target",
    str(goal)
)

col4.metric(
    "Safety Buffer",
    f"{safety_buffer} cells"
)


# ============================================================
# CALCULATE BUTTON
# ============================================================

if st.button(
    "🚀 CALCULATE A* ROUTE",
    use_container_width=True
):

    planning_grid = create_safety_buffer(
        grid,
        safety_buffer
    )

    planning_grid[start] = 0
    planning_grid[goal] = 0

    start_time = time.perf_counter()

    route, nodes = a_star(
        planning_grid,
        start,
        goal
    )

    calculation_time = (
        time.perf_counter()
        - start_time
    ) * 1000

    st.session_state.route = route

    if route is None:

        st.error(
            """
            ❌ No path exists between the start and target.

            Try removing some obstacles or reducing the safety buffer.
            """
        )

    else:

        st.success(
            "✅ Safe path found!"
        )

        # --------------------------
        # DISTANCE
        # --------------------------

        distance = calculate_distance(
            route,
            cell_size
        )

        # --------------------------
        # WIND EFFECT
        # --------------------------

        wind_factor = min(
            0.60,
            0.03 * wind_speed
        )

        effective_speed = (
            drone_speed
            * (1 - wind_factor)
        )

        effective_speed = max(
            drone_speed * 0.25,
            effective_speed
        )

        # --------------------------
        # TIME
        # --------------------------

        flight_time = (
            distance
            / effective_speed
        )

        # --------------------------
        # ENERGY
        # --------------------------

        energy = (
            power
            * flight_time
            / 3600
        )

        # --------------------------
        # BATTERY
        # --------------------------

        battery_wh = (
            battery_capacity
            / 1000
        ) * voltage

        battery_used = (
            energy
            / battery_wh
        ) * 100

        battery_remaining = max(
            0,
            100 - battery_used
        )

        # --------------------------
        # ROUTE MAP
        # --------------------------

        st.subheader(
            "🛩️ Final Flight Path"
        )

        result_fig = draw_map(
            grid,
            start,
            goal,
            route,
            planning_grid
        )

        st.pyplot(
            result_fig,
            use_container_width=True
        )

        plt.close(
            result_fig
        )

        # --------------------------
        # RESULTS
        # --------------------------

        st.subheader(
            "📊 Flight Results"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Distance",
            f"{distance:.2f} m"
        )

        c2.metric(
            "Flight Time",
            f"{flight_time:.2f} s"
        )

        c3.metric(
            "Energy",
            f"{energy:.2f} Wh"
        )

        c4.metric(
            "Battery Remaining",
            f"{battery_remaining:.1f}%"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Nodes Explored",
            nodes
        )

        c2.metric(
            "Calculation Time",
            f"{calculation_time:.3f} ms"
        )

        c3.metric(
            "Path Points",
            len(route)
        )

        # --------------------------
        # MISSION STATUS
        # --------------------------

        if energy <= 0.8 * battery_wh:

            st.success(
                "✅ Mission is feasible with 20% battery reserve."
            )

        else:

            st.warning(
                "⚠️ Estimated energy is too high for the selected battery."
            )

        # --------------------------
        # ENGINEERING CALCULATIONS
        # --------------------------

        st.subheader(
            "🔬 Engineering Calculations"
        )

        st.write(
            f"""
            **Battery Energy**

            E = V × Ah

            E = {voltage:.2f} × {battery_capacity / 1000:.2f}

            E = **{battery_wh:.2f} Wh**
            """
        )

        st.write(
            f"""
            **Distance**

            Route distance = **{distance:.2f} m**
            """
        )

        st.write(
            f"""
            **Flight Time**

            t = D / V

            Estimated flight time = **{flight_time:.2f} s**
            """
        )

        st.write(
            f"""
            **Energy Consumption**

            E = P × t

            Estimated energy = **{energy:.2f} Wh**
            """
        )

        st.write(
            f"""
            **Wind**

            Wind speed = **{wind_speed:.2f} m/s**

            Wind direction = **{wind_direction}°**
            """
        )

        # --------------------------
        # DOWNLOAD REPORT DATA
        # --------------------------

        result_data = pd.DataFrame(
            {
                "Parameter": [
                    "Start",
                    "Target",
                    "Obstacles",
                    "Safety Buffer",
                    "Drone Speed (m/s)",
                    "Battery (mAh)",
                    "Voltage (V)",
                    "Wind Speed (m/s)",
                    "Distance (m)",
                    "Flight Time (s)",
                    "Energy (Wh)",
                    "Battery Remaining (%)",
                    "Nodes Explored"
                ],

                "Value": [
                    str(start),
                    str(goal),
                    int(np.sum(grid == 1)),
                    safety_buffer,
                    drone_speed,
                    battery_capacity,
                    voltage,
                    wind_speed,
                    round(distance, 2),
                    round(flight_time, 2),
                    round(energy, 2),
                    round(battery_remaining, 2),
                    nodes
                ]
            }
        )

        csv_data = result_data.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            data=csv_data,
            file_name="drone_results.csv",
            mime="text/csv"
        )


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

st.markdown("---")

st.subheader(
    "🎓 Project Concept"
)

st.write(
    """
This system is a UAV flight-path planning simulator.
The user manually creates an environment by marking
obstacles on the grid. The A* algorithm then calculates
a path from the selected start point to the target while
maintaining a configurable safety buffer.

The project also estimates flight distance, time,
energy consumption and remaining battery using simplified
engineering models.
"""
)