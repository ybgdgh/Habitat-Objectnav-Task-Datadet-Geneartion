import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.object_nav.objectnav_generator import generate_objectnav_episode, generate_objectnav_goals_by_category

num_episodes_per_scene = int(1e2)

count_scene = 0
def _generate_fn(scene):
    global count_scene
    print("episode start")
    cfg = habitat.get_config(config_paths="configs/datasets/objectnav/gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    # print(vars(sim))
    print(sim.config.SCENE)
    # scene = sim.semantic_annotations()
    # print(len(scene.objects))
    # print(scene.regions)
    # for obj in scene.objects:
    #     if obj is not None:
    #         print(
    #             f"Object id:{obj.id}, category:{obj.category.name()}, index:{obj.category.index()}"
    #             f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
    #         )


    dset = habitat.datasets.make_dataset("ObjectNav-v1")
    # print(vars(dset))
    # print(num_episodes_per_scene)
    dset.goals_by_category = dict(
        generate_objectnav_goals_by_category(
            sim
        )
    )
    print(len(dset.goals_by_category))

    dset.episodes = list(
        generate_objectnav_episode(
            sim, num_episodes_per_scene, is_gen_shortest_path=True
        )
    )
    for ep in dset.episodes:
        ep.scene_id = ep.scene_id[len("./data/scene_datasets/") :]

    dset.category_to_task_category_id = {"kite": 0, "microwave": 1, "oven":2, "sink":3, "refrigerator":4, "clock":5, "toilet":6, "chair":7}
    dset.category_to_scene_annotation_category_id = {"kite": 1, "microwave": 2, "oven":3, "sink":4, "refrigerator":5, "clock":6, "toilet":7, "chair":8}

    scene_key = osp.basename(scene)[:-4]
    print(scene_key)
    out_file = f"./data/datasets/objectnav/gibson/v1/all/content/{scene_key}.json.gz"
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    count_scene=count_scene+1
    print(count_scene)
    print("episode finish!")


scenes = glob.glob("./data/scene_datasets/gibson/*.glb")
print(scenes)
print(len(scenes))
with multiprocessing.Pool(1) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
    for _ in pool.imap_unordered(_generate_fn, scenes):
        pbar.update()

# with gzip.open(f"./data/datasets/objectnav/gibson/v1/all/all.json.gz", "wt") as f:
#     json.dump(dict(episodes=[], 
#         category_to_task_category_id=dset.category_to_task_category_id, 
#         category_to_gibson_category_id=det.category_to_scene_annotation_category_id), f)