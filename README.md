# habitat跨场景室内3D仿真平台生成Gibson环境下的objectnav task datasets

##  

对于已有场景数据集，如何生成自己想要的objectnav task dataset，来满足如不同场景、不同episode次数、不同导航距离（从近到远）等需求，需要自己去生成相应的数据集。这里使用Gibson数据集进行objectnav任务，在进行之前需要先通过[3D-Scene-Graph](https://github.com/StanfordVL/3DSceneGraph)获取Gibson的语义信息，参照https://github.com/facebookresearch/habitat-sim#datasets。

objectnav任务数据集的类为ObjectNavDatasetV1，定义如下：

```python
@registry.register_dataset(name="ObjectNav-v1")
class ObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Object Navigation dataset.
    """
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[ObjectGoalNavEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, List[ObjectGoal]]

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = dict()
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = ObjectGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        # for i in range(len(self.episodes)):
        #     self.episodes[i].goals = self.goals_by_category[
        #         self.episodes[i].goals_key
        #     ]

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.goals_by_category = {}
        super().__init__(config)

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            # view_location = ObjectViewLocation(**view)
            # view_location.agent_state = AgentState(**view_location.agent_state)
            # g.view_points[vidx] = view_location
            g.view_points = []

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if "category_to_scene_annotation_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_scene_annotation_category_id"
            ]

        if "category_to_mp3d_category_id" in deserialized:
            self.category_to_scene_annotation_category_id = deserialized[
                "category_to_mp3d_category_id"
            ]

        assert len(self.category_to_task_category_id) == len(
            self.category_to_scene_annotation_category_id
        )

        assert set(self.category_to_task_category_id.keys()) == set(
            self.category_to_scene_annotation_category_id.keys()
        ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        if "goals_by_category" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        for k, v in deserialized["goals_by_category"].items():
            self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = ObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = self.goals_by_category[episode.goals_key]

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        if point is None or isinstance(point, (int, str)):
                            point = {
                                "action": point,
                                "rotation": None,
                                "position": None,
                            }

                        path[p_index] = ShortestPathPoint(**point)

            self.episodes.append(episode)

```

该类主要由四个部分组成。

### 1. episodes

episodes类型为List[ObjectGoalNavEpisode]，是一个ObjectGoalNavEpisode类组成是数组，包含数据集中一个episode所需要的所有参数：

```python
ObjectGoalNavEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        object_category=object_category,
        shortest_paths=shortest_paths,
        info=info,
)
```

在pointnav任务中，goals参数直接被赋予目标点的位置，而在objectnav中goals为语义目标，数量不一定只有一个，是一个多目标导航，所以这一项为空，对应的是goals_by_category中的数据。在确定一个episode中的起点和终点前，需要测试起是否可到达，其中一个重要条件就是是否为一个楼层。测试函数如下：

```python
def is_compatible_episode(
    s, t, id, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    # check height difference to assure s and  tar are from same floor
    tar = []
    for i in range(len(t)):
        if np.abs(s[1] - t[i][1]) > 0.8:
            continue
        else:
            tar.append(t[i])
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
    closest_goal_id = int(re.findall(r"\d+",id[index])[0])
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
```

t对应多个目标的位姿，与s先进行同一楼层的判定（这里可以讲s的高度确定下来，不需要随机产生），然后再计算欧氏距离，测地距离（规划的避障路径距离），和离s最近的目标id，再对测地距离进行筛选，确定范围和与欧式距离的比例（比例过小说明在一条直线上，过大说明路径过于弯曲）。根据此函数可以设置不同难度的导航任务。

### 2. goals_by_category

goals_by_category类型为Dict[str, List[ObjectGoal]]，str为scene_object数组，和ObjectGoal类组成的列表。str可以根据ObjectGoal中的函数重写，f"{os.path.basename(sim.config.SCENE)}_{object_category}"。ObjectGoal类型定义如下：

```python
@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None
```

其中继承的NavigationGoal类型如下：

```python
@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy.
    """

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None
```

需要注意的是view_points项，该项为在靠近目标附近的位姿和视角中的iou，若在配置文件中的DISTANCE_TO: VIEW_POINTS，则使用这些view_points来计算距离，否则使用object的位置来计算距离。由于view_points不易生成，因此这里没有使用。

这里每一个ObjectGoal都表示str场景中的object中的一个，通过list将同一类object放在一起，然后通过dict将同一场景下的object放到一起，方便检索。

### 3. category_to_task_category_id

该类型表示在任务中排序的类别id，如：

```python
category_to_task_category_id = {"kite": 0, "microwave": 1, "oven":2, "sink":3, "refrigerator":4, "clock":5, "toilet":6, "chair":7}
```

### 4. category_to_scene_annotation_category_id

该类型表示在数据集中排序的类别id，如：

```python
category_to_scene_annotation_category_id = {"kite": 1, "microwave": 2, "oven":3, "sink":4, "refrigerator":5, "clock":6, "toilet":7, "chair":8}
```

## Use

```sh
python create_objectnav_dataset.py
```

## Dataset

使用gibson_habitat_trainval生成的78个语义场景，训练集中每个场景中的goal设置生成1000个episode用于训练，测试集中每个goal设置生成10个episode用于测试，这里的goal根据在Gibson数据集中出现的次数，选择了如下21种出现次数最多的类别：

```python
"chair",
"potted plant",
"sink",
"vase",
"book",
"couch",
"bed",
"bottle",
"dining table",
"toilet",
"refrigerator",
"tv",
"clock",
"oven",
"bowl",
"cup",
"bench",
"microwave",
"suitcase",
"umbrella",
"teddy bear"
```

Download：[Gibson-objectnav-datasets](https://drive.google.com/file/d/1vypkzXIfGyN42rAjOzjny8AeFaRpWLZW/view?usp=sharing)







