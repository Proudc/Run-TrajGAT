import yaml


if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)

    config["data"] = "CCCCCCCCCCCCCCCCCCCCC"

    print(config)
