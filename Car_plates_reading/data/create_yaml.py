import yaml

def create_yaml(yolo_dir: str, classes: list) -> None:
   
    yaml_file = yolo_dir + '/data/'

    yaml_data = dict(
        path = yaml_file + "plates_detection/",
        train = "train",
        val = "val",
        nc = len(classes),
        names = classes
    )

    with open(yaml_file + "plates.yaml", 'w') as f:
        yaml.dump(yaml_data, f, explicit_start = True, default_flow_style = False)

    print(".yaml file created.")