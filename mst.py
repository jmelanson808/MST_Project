import heapq
import math
import sys
import time

import matplotlib.pyplot as plt
import geopandas as gpd
import bridges


def get_mst(cities, algorithm):
    t0 = time.time()
    names = list(cities.keys())
    coords = list(cities.values())

    mst = {}
    visited = []
    heap = []

    visited.append(names[0])

    if algorithm == "prim1":
        mst[names[0]] = {"position": coords[0], "parent": coords[0]}

        while len(visited) < len(cities):
            shortest_dist = float('inf')
            best_city_index = None
            best_parent_index = None

            # Check all unvisited nodes.
            for city_index, city_name in enumerate(names):
                if city_name in visited:
                    continue

                for parent_index, parent_name in enumerate(visited):
                    distance = get_distance(coords[city_index], coords[names.index(parent_name)])
                    if distance < shortest_dist:
                        shortest_dist = distance
                        best_city_index = city_index
                        best_parent_index = names.index(parent_name)

            # Add the shortest edge to the MST and visited.
            visited.append(names[best_city_index])
            mst[names[best_city_index]] = {
                "position": coords[best_city_index],
                "parent": coords[best_parent_index]
            }

        print('Edges:', len(mst))

    elif algorithm == 'prim2':
        min_dist = {}
        # Load all nodes adjacent to starting node to the heap.
        for i in range(1, len(cities)):
            distance = get_distance(coords[i], coords[0])
            heapq.heappush(heap, (distance, i, 0))  # (priority, city_index, parent_index)
            min_dist[names[i]] = (distance, i, 0)

        # Check each node popped off the heap.
        while len(visited) < len(cities):
            priority, city_index, parent_index = heapq.heappop(heap)

            if names[city_index] in visited:
                continue

            # Add unvisited node to visited and MST.
            visited.append(names[city_index])
            mst[names[city_index]] = {
                "position": coords[city_index],
                "parent": coords[parent_index]
            }

            # Load all adjacent nodes to the heap.
            for i in range(len(cities)):
                if names[i] not in visited:
                    distance = get_distance(coords[i], coords[city_index])
                    if distance > 0:
                        if distance < min_dist[names[i]][0]:
                            min_dist[names[i]] = (distance, i, city_index)
                            heapq.heappush(heap, (distance, i, city_index))

            print('Heap:', len(heap), '|', 'Edges:', len(mst))

    elif algorithm == "kruskal":
        edges = []
        parents = list(range(len(cities)))
        rank = [0] * len(cities)
        mst = []

        for i in range(len(cities)):
            for j in range(len(cities)):
                if i != j:
                    edges.append(((i, j), get_distance(coords[i], coords[j])))

        # Sort the edges by distance in ascending order.
        sorted_edges = sorted(edges, key=lambda x: x[1])

        # Iterate through all edge possibilities and add to MST if vertices from different root parents.
        for (a, b), weight in sorted_edges:
            if find(parents, a) != find(parents, b):
                mst.append({
                    "position": coords[b],
                    "parent": coords[a]
                })
                union(parents, rank, a, b)

        print('Edges:', len(mst))

    else:
        print("Invalid algorithm.")
        return

    t1 = time.time()
    print('Program took', round(t1 - t0, 2), 'seconds to complete.')

    return Result(mst, round(t1 - t0, 2))


# Calculates distance between two points.
def get_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# Recursively searches up the subtree to find i's root parent.
def find(parents, i):
    if parents[i] == i:
        return i    # i is its own root parent if not connected to any other group.
    parents[i] = find(parents, parents[i])
    return parents[i]


# Uses the ranked union technique to incorporate a and b into the same subtree.
# This ensures a cycle isn't created while iterating through the sorted edges.
def union(parents, rank, a, b):
    rootA = find(parents, a)
    rootB = find(parents, b)

    if rootA != rootB:
        if rank[rootA] > rank[rootB]:
            parents[rootB] = rootA # b's parent is now rootA
        elif rank[rootA] < rank[rootB]:
            parents[rootA] = rootB # a's parent is now rootB
        else:
            parents[rootB] = rootA # b's parent is now rootA
            rank[rootA] += 1 # increment union rank


# Uses Matplotlib and Geopandas to draw the tree.
def display_MST(graph, state, algorithm, min_pop):
    us_states = gpd.read_file("US_state_shapefile/tl_2024_us_state.shp")

    fig, ax = plt.subplots(figsize=(16, 12))

    if state == "United States":
        us_states.plot(ax=ax, edgecolor="black", facecolor="white", linewidth=0.25)
    else:
        state_outline = us_states[us_states["NAME"] == state]
        state_outline.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5)

    if algorithm == "kruskal":
        for edge in graph.mst:
            location = edge["position"]
            source = edge["parent"]
            ax.plot([source[0], location[0]],
                    [source[1], location[1]],
                    marker='o',
                    markersize=2,
                    linewidth=1,
                    color="red"
                    )
    else:
        for city, coords in graph.mst.items():
            location = coords["position"]
            source = coords["parent"]
            ax.plot([source[0], location[0]],
                    [source[1], location[1]],
                    marker='o',
                    markersize=2,
                    linewidth=1,
                    color="red"
                    )

    if state == "United States":
        plt.xlim(-160, -60)
        plt.ylim(15, 65)
    if state == "Alaska":
        plt.xlim(-190, -125)
        plt.ylim(50, 72.5)
    if state == "Hawaii":
        plt.xlim(-161, -154)
        plt.ylim(18.5, 22.5)

    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title('Minimum Spanning Tree of US Cities in ' + state + ' (pop. ' + min_pop + '+) using ' + algorithm + ' (' + str(graph.duration) + ' seconds)')

    plt.show()


# Maps state parameter so Geopandas and Bridges plays nice together.
def state_mapper(state):
    state_names = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
        "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
        "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
        "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "US": "United States", "UT": "Utah",
        "VA": "Virginia", "VT": "Vermont", "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming"
    }

    return state_names[state]


class MST:
    def __init__(self, state, alg, min_pop):
        # Build city data based on state parameter.
        self.min_pop = int(min_pop)

        if state == "US":
            self.data = filter(lambda c: c.population > self.min_pop, bridges.get_us_cities_data())
        else:
            self.data = filter(lambda c: c.state == state and c.population > self.min_pop, bridges.get_us_cities_data())

        self.cities = {}
        self.test_cities = {}
        self.state = state
        self.alg = alg
        self.time = time.time()


        # We only need the geographic coordinates for each city.
        for city in self.data:
            self.cities[city.city] = (city.lon, city.lat)


    def solve_MST(self, algorithm, min_pop):
        result = get_mst(self.cities, algorithm)
        display_MST(result, state_mapper(self.state), self.alg, min_pop)


class Result:
    def __init__(self, mst, duration):
        self.mst = mst
        self.duration = duration


if __name__ == '__main__':
    # Run this program in the terminal like:
    # python mst.py UT prim1 5000
    # You can substitute UT parameter with any other US state abbreviation.
    #   OR you can enter US to see the MST for all US cities with population over 250000.
    # Options for second parameter are prim1, prim2, or kruskal.
    # The third parameter will set the minimum population for a city to be used in the graph.
    tree = MST(sys.argv[1], sys.argv[2], sys.argv[3])
    tree.solve_MST(sys.argv[2], sys.argv[3])
