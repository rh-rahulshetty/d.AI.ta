import json
from typing import List
from src.modals.file_types.json_data import JSONFileMetadata, JSONKey, JSONType
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)

MAX_ITEM_LOOK_UP_FOR_TYPE = 100


def create_json_metadata(_id: str, file_path: str) -> JSONFileMetadata:
    '''Generate metadata for JSON file'''
    json_data = None

    with open(file_path, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp)

    json_keys = generate_json_keys_for_obj(json_data)

    return JSONFileMetadata(
        id=_id,
        file_path=file_path,
        json_keys=json_keys,
    )


def generate_json_keys_for_obj(obj: dict | list) -> List[JSONKey]:
    '''Flattens json data'''
    result_keys = []

    queue = [(obj, "")]

    # BFS
    while len(queue) != 0:
        data, key = queue.pop(0)
        if isinstance(data, dict):
            root_key = key + "."
            # result_keys.append(JSONKey(key=root_key, type=JSONType.json))
            for k, v in data.items():
                queue.append(
                    (v, root_key + k)
                )

        elif isinstance(data, list):
            new_key = key + "[]"
            # result_keys.append(JSONKey(key=new_key, type=JSONType.array))
            for item in obj[:MAX_ITEM_LOOK_UP_FOR_TYPE]:
                queue.append((item, new_key))

        elif isinstance(data, int) or isinstance(data, float):
            result_keys.append(JSONKey(key=key, type=JSONType.number))
        elif isinstance(data, str):
            result_keys.append(JSONKey(key=key, type=JSONType.string))
        elif isinstance(data, bool):
            result_keys.append(JSONKey(key=key, type=JSONType.boolean))
    
    return list(set(result_keys))
