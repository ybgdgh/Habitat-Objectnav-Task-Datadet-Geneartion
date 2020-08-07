#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import re

from habitat.core.simulator import Simulator
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ObjectGoal
from habitat_sim.errors import GreedyFollowerError

r"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""
ISLAND_RADIUS_LIMIT = 1.5

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, id, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    # check height difference to assure s and  tar are from same floor
    tar = []
    tar_id = []
    for i in range(len(t)):
        if np.abs(s[1] - t[i][1]) > 0.5:
            continue
        else:
            tar.append(t[i])
            tar_id.append(id[i])
    if len(tar) == 0:
        return False, 0, 0, 0, 0

    euclid_dist_arr = []
    index = 0
    closest_goal_id = 0
    for i in range(len(tar)):
        euclid_dist_arr.append(np.power(np.power(np.array(s) - np.array(tar[i]), 2).sum(0), 0.5))
    euclid_dist = min(euclid_dist_arr)
    index = euclid_dist_arr.index(min(euclid_dist_arr))
    target_position_episode = tar[index]
    # print(id[index], type(id[index]))
    closest_goal_id = int(re.findall(r"\d+",tar_id[index])[0])
    # print(euclid_dist_arr)
    # print(euclid_dist)
    # print(id, closest_goal_id))

    d_separation = sim.geodesic_distance(s, tar)
    if d_separation == np.inf:
        return False, 0, 0, 0, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0, 0, 0, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0, 0, 0, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0, 0, 0, 0
    return True, d_separation, euclid_dist, closest_goal_id, target_position_episode


def _create_episode(
    episode_id,
    scene_id,
    start_position,
    start_rotation,
    target_position,
    object_category,
    shortest_paths=None,
    radius=None,
    info=None,
) -> Optional[ObjectGoalNavEpisode]:
    goals = []
    return ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        object_category=object_category,
        shortest_paths=shortest_paths,
        info=info,
    )


def generate_objectnav_episode(
    sim: Simulator,
    task_category,
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 1000,
) -> ObjectGoalNavEpisode:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    scene = sim.semantic_annotations()
    print("scene object len: ", len(scene.objects))
    target = dict()
    for obj in scene.objects:
        if obj is not None:
            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()}, Index:{obj.category.index()}"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            # )
            if obj.category.name() in task_category.keys():
                if obj.category.index() in target:
                    target[obj.category.index()].append(obj)
                else:
                    target[obj.category.index()] = [obj]
    print("target len:", len(target))

    for i in target:
        print("target episode len:", len(target[i]))
        object_category = target[i][0].category.name()
        print("object_category :", object_category)

        target_position = []
        target_id = []
        for j in range(len(target[i])):
            target_position.append(target[i][j].aabb.center)
            target_id.append(target[i][j].id)
        # print("target_position :", target_position)

        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            # source_position[1] = High

            is_compatible, dist, euclid, closest_goal_object_id, target_position_episode = is_compatible_episode(
                source_position,
                target_position,
                target_id,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )

            if is_compatible:
                break

        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position_episode,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

            episode = _create_episode(
                episode_id=i,
                scene_id=sim.config.SCENE,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                object_category=object_category,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist, "euclidean_distance": euclid, "closest_goal_object_id": closest_goal_object_id},
            )
            print("source_position: ", source_position)
            print("episode finish!")
            yield episode

        else:
            continue


def generate_objectnav_goals_by_category(
    sim: Simulator,
    task_category
) -> ObjectGoal:
    
    scene = sim.semantic_annotations()
    target = dict()
    for obj in scene.objects:
        if obj is not None:
            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()}, Index:{obj.category.index()}"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            # )
            if obj.category.name() in task_category.keys():
                if obj.category.index() in target:
                    target[obj.category.index()].append(obj)
                else:
                    target[obj.category.index()] = [obj]

    for i in target:
        print("target episode len:", len(target[i]))
        object_category = target[i][0].category.name()
        print("object_category :", object_category)
        str_goal = f"{os.path.basename(sim.config.SCENE)}_{object_category}"
        print(str_goal)

        goals_by = []
        for j in range(len(target[i])):
            goal_by_object = ObjectGoal(
                position = target[i][j].aabb.center,
                radius = 0.5,
                object_id = int(re.findall(r"\d+",target[i][j].id)[0]),
                object_name = target[i][j].id,
                object_category = object_category,
                view_points = [],
            )
        
            goals_by.append(goal_by_object)       

        yield str_goal, goals_by

def generate_objectnav_task_category_id(
    sim: Simulator,
    task_category
):
    scene = sim.semantic_annotations()
    target = dict()
    for obj in scene.objects:
        if obj is not None:
            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()}, Index:{obj.category.index()}"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            # )
            if obj.category.name() in task_category.keys():
                if obj.category.index() in target:
                    target[obj.category.index()].append(obj)
                else:
                    target[obj.category.index()] = [obj]
    task_category_id = task_category
    for i in target:
        task_category_id[target[i][0].category.name()] = target[i][0].category.index()
    print(task_category_id)
    return task_category_id